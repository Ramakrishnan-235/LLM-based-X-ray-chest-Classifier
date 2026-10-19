import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent Windows console UnicodeEncodeError for emojis like MLflow's runner emoji
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
import yaml

from services.inference.classification.calibration import (
    TemperatureScaler,
    compute_ece,
)


TARGET_FINDINGS = {
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Pneumothorax": "Pneumothorax",
    "Atelectasis": "Atelectasis",
    "Lung Opacity": "Lung Opacity",
}


def generate_synthetic_cohort(n_patients: int = 800, seed: int = 42):
    """
    Simulates uncalibrated (overconfident) model logits and ground-truth labels
    grouped by patient_id to verify patient-level splitting and MLflow logging.
    """
    rng = np.random.default_rng(seed)
    n_studies = n_patients * 2
    patient_ids = np.repeat(np.arange(n_patients), 2)

    logits_dict, labels_dict = {}, {}
    true_temps = {
        "Cardiomegaly": 1.35,
        "Effusion": 1.25,
        "Pneumothorax": 1.45,
        "Atelectasis": 1.20,
        "Lung Opacity": 1.30,
    }
    prevalences = {
        "Cardiomegaly": 0.18,
        "Effusion": 0.24,
        "Pneumothorax": 0.09,
        "Atelectasis": 0.22,
        "Lung Opacity": 0.30,
    }

    for k in TARGET_FINDINGS:
        y = rng.binomial(1, prevalences[k], size=n_studies).astype(np.float32)
        # Generate discriminative but overconfident raw logits
        base_z = np.where(y == 1, rng.normal(1.2, 1.4, n_studies), rng.normal(-1.8, 1.4, n_studies))
        logits_dict[k] = (base_z * true_temps[k]).astype(np.float32)
        labels_dict[k] = y

    return patient_ids, logits_dict, labels_dict


def plot_reliability_diagrams(
    test_logits: dict,
    test_labels: dict,
    calibrated_cfg: dict,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for idx, (finding_key, ui_label) in enumerate(TARGET_FINDINGS.items()):
        ax = axes[idx]
        z = test_logits[finding_key]
        y = test_labels[finding_key]
        temp = calibrated_cfg[finding_key]["temperature"]

        raw_p = expit(z)
        cal_p = expit(z / temp)

        prob_true_raw, prob_pred_raw = calibration_curve(y, raw_p, n_bins=8)
        prob_true_cal, prob_pred_cal = calibration_curve(y, cal_p, n_bins=8)

        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#ef4444", label="Uncalibrated")
        ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#10b981", label=f"Temp Scaled (T={temp})")
        ax.set_title(ui_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Score")
        if idx == 0:
            ax.set_ylabel("Observed Fraction Positive")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def run_evaluation(tracking_uri: str, experiment_name: str, update_yaml: bool = False):
    # Configure S3/MinIO credentials for MLflow artifact uploads
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio_admin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio_local_secret")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    patient_ids, logits_dict, labels_dict = generate_synthetic_cohort(n_patients=1000)

    # Strict Patient-Level Split (50% Calibration Validation / 50% Held-out Evaluation)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss.split(np.zeros(len(patient_ids)), groups=patient_ids))

    # Assert zero patient overlap between calibration and test splits
    overlap = set(patient_ids[val_idx]).intersection(set(patient_ids[test_idx]))
    assert len(overlap) == 0, f"CRITICAL: Patient leakage detected ({len(overlap)} overlapping IDs)!"

    val_logits = {k: v[val_idx] for k, v in logits_dict.items()}
    val_labels = {k: v[val_idx] for k, v in labels_dict.items()}
    test_logits = {k: v[test_idx] for k, v in logits_dict.items()}
    test_labels = {k: v[test_idx] for k, v in labels_dict.items()}

    # Fit Temperature Scaling and 3-Zone Thresholds on Validation Split
    calibrated_cfg = TemperatureScaler.fit_all_findings(
        val_logits, val_labels, TARGET_FINDINGS
    )

    with mlflow.start_run(run_name="densenet121-baseline-calibration"):
        mlflow.log_param("backbone", "densenet121-res224-all")
        mlflow.log_param("patient_split_method", "GroupShuffleSplit")
        mlflow.log_param("val_patients", len(set(patient_ids[val_idx])))
        mlflow.log_param("test_patients", len(set(patient_ids[test_idx])))

        summary_metrics = {}
        for finding_key, ui_label in TARGET_FINDINGS.items():
            z_test = test_logits[finding_key]
            y_test = test_labels[finding_key]
            cfg = calibrated_cfg[finding_key]

            raw_probs = expit(z_test)
            cal_probs = expit(z_test / cfg["temperature"])

            auroc = float(roc_auc_score(y_test, cal_probs))
            auprc = float(average_precision_score(y_test, cal_probs))
            brier_raw = float(brier_score_loss(y_test, raw_probs))
            brier_cal = float(brier_score_loss(y_test, cal_probs))
            ece_raw = compute_ece(raw_probs, y_test)
            ece_cal = compute_ece(cal_probs, y_test)
            f1_high = float(f1_score(y_test, (cal_probs >= cfg["high_threshold"]).astype(int)))

            slug = finding_key.lower().replace(" ", "_")
            mlflow.log_metrics(
                {
                    f"{slug}_auroc": round(auroc, 4),
                    f"{slug}_auprc": round(auprc, 4),
                    f"{slug}_brier_pre_cal": round(brier_raw, 4),
                    f"{slug}_brier_post_cal": round(brier_cal, 4),
                    f"{slug}_ece_pre_cal": round(ece_raw, 4),
                    f"{slug}_ece_post_cal": round(ece_cal, 4),
                    f"{slug}_f1_at_high_thresh": round(f1_high, 4),
                    f"{slug}_temperature": cfg["temperature"],
                }
            )
            summary_metrics[ui_label] = {
                "AUROC": round(auroc, 4),
                "AUPRC": round(auprc, 4),
                "ECE (Pre -> Post)": f"{ece_raw:.4f} -> {ece_cal:.4f}",
                "Brier (Pre -> Post)": f"{brier_raw:.4f} -> {brier_cal:.4f}",
                "Thresholds (Low / High)": f"{cfg['low_threshold']} / {cfg['high_threshold']}",
            }

        # Save artifacts and upload to MLflow (MinIO S3)
        artifact_dir = Path("ml/experiments/latest_artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cal_json_path = artifact_dir / "calibration_artifact.json"
        with open(cal_json_path, "w", encoding="utf-8") as f:
            json.dump(calibrated_cfg, f, indent=2)

        plot_path = artifact_dir / "reliability_diagrams.png"
        plot_reliability_diagrams(test_logits, test_labels, calibrated_cfg, plot_path)

        mlflow.log_artifact(str(cal_json_path))
        mlflow.log_artifact(str(plot_path))

        if update_yaml:
            yaml_path = Path("ml/configs/findings_scope.yaml")
            with open(yaml_path, "r", encoding="utf-8") as f:
                scope_doc = yaml.safe_load(f)
            scope_doc["findings"] = calibrated_cfg
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(scope_doc, f, sort_keys=False)

        print("\n=== Sprint 2 Baseline Calibration & Evaluation Summary ===")
        print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment-name", default="cxr-insight-stage1-baseline")
    parser.add_argument("--update-yaml", action="store_true")
    args = parser.parse_args()

    run_evaluation(args.tracking_uri, args.experiment_name, args.update_yaml)
