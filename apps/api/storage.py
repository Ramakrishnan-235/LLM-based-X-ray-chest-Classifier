import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from apps.api.config import settings


class StorageService:
    """Manages private S3/MinIO object storage and pre-signed URLs."""

    def __init__(self):
        self.bucket = settings.minio_bucket_name
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_root_user,
            aws_secret_access_key=settings.minio_root_password,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def generate_presigned_upload_url(
        self, object_key: str, content_type: str = "application/dicom", expires_in: int = 900
    ) -> str:
        return self.s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": self.bucket,
                "Key": object_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires_in,
        )

    def generate_presigned_download_url(
        self, object_key: str, expires_in: int = 3600
    ) -> str:
        return self.s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )

    def object_exists(self, object_key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except ClientError:
            return False

    def get_object_bytes(self, object_key: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=object_key)
        return response["Body"].read()

    def put_object_bytes(
        self, object_key: str, data: bytes, content_type: str = "image/png"
    ) -> str:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        return object_key


storage_service = StorageService()
