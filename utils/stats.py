from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def compute_mean_std(values: List[float]) -> Dict[str, float]:
    """Computes sample mean and sample standard deviation."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def paired_significance_test(
    scores_method: List[float],
    scores_baseline: List[float],
) -> Dict[str, float]:
    """
    Performs paired statistical testing between method and baseline.
    Employs Wilcoxon signed-rank test where valid; falls back to paired t-test.
    """
    a = np.array(scores_method, dtype=np.float64)
    b = np.array(scores_baseline, dtype=np.float64)

    if len(a) != len(b) or len(a) < 2:
        return {"p_value": 1.0, "significant": False, "test": "insufficient_samples"}

    diff = a - b
    if np.all(diff == 0):
        return {"p_value": 1.0, "significant": False, "test": "identical"}

    try:
        stat, p_val = stats.wilcoxon(a, b, zero_method="wilcox", correction=True)
        test_name = "wilcoxon"
    except Exception:
        stat, p_val = stats.ttest_rel(a, b)
        test_name = "paired_ttest"

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "test": test_name,
    }


def bootstrap_ci_95(values: List[float], n_bootstraps: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Computes 95% bootstrap confidence interval [ci_lower, ci_upper].
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) < 2:
        val = float(arr[0]) if len(arr) == 1 else 0.0
        return val, val

    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return ci_lower, ci_upper
