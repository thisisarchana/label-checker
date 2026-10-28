"""Rewrites Open Food Facts allergen labels to follow the gold-set labeling policy."""
import pandas as pd

from ml.build_training_data import PEANUT_OIL, SOY_OIL, oil_only


def apply_policy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Soy or peanut that appears only as 'soybean oil' / 'peanut oil' is not labeled."""
    out = df.copy()
    out.loc[out.text.apply(lambda t: oil_only(t, SOY_OIL, "soy")), "soy"] = 0
    out.loc[out.text.apply(lambda t: oil_only(t, PEANUT_OIL, "peanut")), "peanuts"] = 0
    return out