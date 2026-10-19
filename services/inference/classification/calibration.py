from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_curve


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins equal-width probability bins.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs > bin_lower) & (probs <= bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler:
    """Fits per-class temperature scaling and derives 3-zone clinical thresholds."""

    @staticmethod
    def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
        """Minimizes binary cross-entropy (NLL) w.r.t. scalar temperature T > 0."""
        valid_mask = ~np.isnan(labels) & (labels >= 0)
        z = logits[valid_mask].astype(np.float64)
        y = labels[valid_mask].astype(np.float64)

        def nll_loss(t_array: np.ndarray) -> float:
            t = max(float(t_array[0]), 1e-3)
            p = np.clip(expit(z / t), 1e-7, 1.0 - 1e-7)
            return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        res = minimize(
            nll_loss,
            x0=np.array([1.0]),
            bounds=[(0.1, 10.0)],
            method="L-BFGS-B",
        )
        return round(float(res.x[0]), 4)

    @staticmethod
    def derive_three_zone_thresholds(
        calibrated_probs: np.ndarray,
        labels: np.ndarray,
        target_sensitivity_floor: float = 0.85,
    ) -> Tuple[float, float]:
        """
        Returns (low_threshold, high_threshold):
        - low_threshold: threshold achieving >= target_sensitivity_floor (rule-out boundary)
        - high_threshold: threshold maximizing Youden's J = Sensitivity + Specificity - 1
        """
        valid_mask = ~np.isnan(labels) & (labels >= 0)
        p = calibrated_probs[valid_mask]
        y = labels[valid_mask]

        fpr, tpr, thresholds = roc_curve(y, p)
        # Filter out inf threshold inserted by sklearn at index 0
        finite_mask = np.isfinite(thresholds)
        fpr, tpr, thresholds = fpr[finite_mask], tpr[finite_mask], thresholds[finite_mask]

        # 1. High threshold via Youden's J
        youden_j = tpr - fpr
        best_j_idx = int(np.argmax(youden_j))
        high_thresh = float(np.clip(thresholds[best_j_idx], 0.50, 0.90))

        # 2. Low threshold via sensitivity floor (tpr >= target_sensitivity_floor)
        sens_indices = np.where(tpr >= target_sensitivity_floor)[0]
        if len(sens_indices) > 0:
            # First index where sensitivity reaches target (highest threshold meeting floor)
            low_candidate = float(thresholds[sens_indices[0]])
        else:
            low_candidate = high_thresh * 0.65

        # Guarantee strict separation between low_threshold and high_threshold
        low_thresh = float(np.clip(min(low_candidate, high_thresh - 0.12), 0.20, 0.75))
        return round(low_thresh, 4), round(high_thresh, 4)

    @classmethod
    def fit_all_findings(
        cls,
        logits_dict: Dict[str, np.ndarray],
        labels_dict: Dict[str, np.ndarray],
        ui_labels: Dict[str, str],
    ) -> Dict[str, dict]:
        """Fits T, low_threshold, and high_threshold across all 5 target findings."""
        calibrated_config: Dict[str, dict] = {}
        for finding_key, z in logits_dict.items():
            y = labels_dict[finding_key]
            temp = cls.fit_temperature(z, y)
            cal_p = expit(z / temp)
            low_t, high_t = cls.derive_three_zone_thresholds(cal_p, y)

            calibrated_config[finding_key] = {
                "ui_label": ui_labels.get(finding_key, finding_key),
                "temperature": temp,
                "low_threshold": low_t,
                "high_threshold": high_t,
            }
        return calibrated_config
