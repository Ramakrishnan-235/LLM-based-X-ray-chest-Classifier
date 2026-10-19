import io
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from services.inference.preprocessing.pipeline import CXRPreprocessor


def create_synthetic_dicom(
    rows: int = 512,
    cols: int = 512,
    photometric: str = "MONOCHROME2",
    view_position: str = "PA",
    add_burned_in_banner: bool = False,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Add synthetic PHI that must be stripped
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "PHI-998877"

    # Technical metadata
    ds.Modality = "DX"
    ds.ViewPosition = view_position
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0

    # Generate synthetic gradient chest radiograph (uint16)
    rng = np.random.default_rng(42)
    img = rng.integers(100, 3500, size=(rows, cols), dtype=np.uint16)
    # Central dark lung-like region
    img[100:400, 100:400] = 300

    if add_burned_in_banner:
        # Simulate bright burned-in text banner at top border
        img[10:28, 50:220] = 4095

    ds.PixelData = img.tobytes()

    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, write_like_original=False)
    return buffer.getvalue()


def test_monochrome2_pa_dicom_accepted():
    preprocessor = CXRPreprocessor()
    dcm_bytes = create_synthetic_dicom(
        photometric="MONOCHROME2", view_position="PA", add_burned_in_banner=True
    )
    res = preprocessor.process_bytes(dcm_bytes, filename="sample.dcm")

    assert res.qc_status == "accepted"
    assert res.view_position == "PA"
    assert res.photometric_inverted is False
    assert res.border_phi_masked is True
    assert res.tensor_input is not None
    assert res.tensor_input.shape == (1, 1, 224, 224)
    assert float(np.min(res.tensor_input)) >= -1024.0
    assert float(np.max(res.tensor_input)) <= 1024.0
    # Ensure PHI was stripped from metadata
    assert "PatientName" not in res.safe_metadata
    assert "PatientID" not in res.safe_metadata


def test_monochrome1_inversion():
    preprocessor = CXRPreprocessor()
    dcm_bytes = create_synthetic_dicom(photometric="MONOCHROME1", view_position="AP")
    res = preprocessor.process_bytes(dcm_bytes, filename="inverted.dcm")

    assert res.qc_status == "accepted"
    assert res.photometric_inverted is True
    assert res.tensor_input.shape == (1, 1, 224, 224)


def test_lateral_view_flagged():
    preprocessor = CXRPreprocessor()
    dcm_bytes = create_synthetic_dicom(view_position="LATERAL")
    res = preprocessor.process_bytes(dcm_bytes, filename="lateral.dcm")

    assert res.qc_status == "flagged_unsupported_view"
    assert res.view_position == "LATERAL"


def test_undersized_image_rejected():
    preprocessor = CXRPreprocessor()
    dcm_bytes = create_synthetic_dicom(rows=128, cols=128)
    res = preprocessor.process_bytes(dcm_bytes, filename="small.dcm")

    assert res.qc_status == "rejected_low_quality"
    assert res.tensor_input is None


def test_standard_image_accepted():
    from PIL import Image

    preprocessor = CXRPreprocessor()
    # Create synthetic 256x256 grayscale image
    arr = np.random.randint(50, 200, size=(256, 256), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res = preprocessor.process_bytes(png_bytes, filename="sample.png")
    assert res.qc_status == "accepted"
    assert res.view_position == "UNKNOWN"
    assert res.tensor_input is not None
    assert res.tensor_input.shape == (1, 1, 224, 224)
    assert res.safe_metadata.get("source_format") == "raster_image"


def test_unsupported_modality_rejected():
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.Modality = "CT"
    ds.Rows = 256
    ds.Columns = 256
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((256, 256), dtype=np.uint16).tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    dcm_bytes = buf.getvalue()

    preprocessor = CXRPreprocessor()
    res = preprocessor.process_bytes(dcm_bytes, filename="ct.dcm")
    assert res.qc_status == "rejected_low_quality"
    assert "Unsupported DICOM Modality: CT" in (res.rejection_reason or "")


def test_near_zero_variance_rejected():
    preprocessor = CXRPreprocessor()
    # Synthetic flat image with zero variance
    arr = np.ones((256, 256), dtype=np.uint8) * 128
    from PIL import Image
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = preprocessor.process_bytes(buf.getvalue(), filename="flat.png")
    assert res.qc_status == "rejected_low_quality"
    assert "Near-zero pixel variance" in (res.rejection_reason or "")

