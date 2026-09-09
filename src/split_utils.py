# ============================================================================
# Group-Aware, Temporal, and Stratified Split Utilities
# Enforces strict isolation: zero group overlap across Train, Validation, and Test
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class SplitRegime(str, Enum):
    IID_HOLDOUT = "IID_HOLDOUT"
    GROUP_HOLDOUT = "GROUP_HOLDOUT"
    TEMPORAL_HOLDOUT = "TEMPORAL_HOLDOUT"


class GroupLeakageError(ValueError):
    """Raised when group identifiers cross split boundaries."""
    pass


class TemporalLeakageError(ValueError):
    """Raised when future information leaks into training splits."""
    pass


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    s_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    s_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    s_test: pd.Series
    regime: SplitRegime
    train_groups: Optional[Set[Any]] = None
    val_groups: Optional[Set[Any]] = None
    test_groups: Optional[Set[Any]] = None

    def verify_isolation(self) -> None:
        """Verifies zero overlap between train, val, and test groups."""
        if self.train_groups is not None and self.val_groups is not None:
            inter_tv = self.train_groups.intersection(self.val_groups)
            if inter_tv:
                raise GroupLeakageError(f"Train/Validation group leakage detected: {len(inter_tv)} overlapping groups.")
        
        if self.train_groups is not None and self.test_groups is not None:
            inter_tt = self.train_groups.intersection(self.test_groups)
            if inter_tt:
                raise GroupLeakageError(f"Train/Test group leakage detected: {len(inter_tt)} overlapping groups.")
                
        if self.val_groups is not None and self.test_groups is not None:
            inter_vt = self.val_groups.intersection(self.test_groups)
            if inter_vt:
                raise GroupLeakageError(f"Validation/Test group leakage detected: {len(inter_vt)} overlapping groups.")

    def summary(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "train_size": len(self.y_train),
            "val_size": len(self.y_val),
            "test_size": len(self.y_test),
            "train_groups_count": len(self.train_groups) if self.train_groups else None,
            "val_groups_count": len(self.val_groups) if self.val_groups else None,
            "test_groups_count": len(self.test_groups) if self.test_groups else None,
        }


def group_aware_3way_split(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    sensitive: Union[pd.Series, np.ndarray],
    groups: Union[pd.Series, np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """
    Partitions dataset by grouping keys so no group crosses split boundaries.
    Used for trip UUID (Delhivery) or station code (Amazon).
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    group_series = pd.Series(groups).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)
    s_series = pd.Series(sensitive).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    unique_groups = group_series.unique()
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    if n_groups < 3:
        raise ValueError(f"At least 3 distinct groups are required for group-aware 3-way split; found {n_groups}.")

    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio)))
    if n_train + n_val >= n_groups:
        n_val = max(1, n_groups - n_train - 1)

    train_grp = set(shuffled_groups[:n_train])
    val_grp = set(shuffled_groups[n_train : n_train + n_val])
    test_grp = set(shuffled_groups[n_train + n_val :])

    # Strict isolation verification
    if train_grp.intersection(val_grp) or train_grp.intersection(test_grp) or val_grp.intersection(test_grp):
        raise GroupLeakageError("Group overlap detected during partition calculation!")

    train_mask = group_series.isin(train_grp).to_numpy()
    val_mask = group_series.isin(val_grp).to_numpy()
    test_mask = group_series.isin(test_grp).to_numpy()

    res = SplitResult(
        X_train=X_df[train_mask].copy(),
        y_train=y_series[train_mask].copy(),
        s_train=s_series[train_mask].copy(),
        X_val=X_df[val_mask].copy(),
        y_val=y_series[val_mask].copy(),
        s_val=s_series[val_mask].copy(),
        X_test=X_df[test_mask].copy(),
        y_test=y_series[test_mask].copy(),
        s_test=s_series[test_mask].copy(),
        regime=SplitRegime.GROUP_HOLDOUT,
        train_groups=train_grp,
        val_groups=val_grp,
        test_groups=test_grp,
    )
    res.verify_isolation()
    return res


def temporal_3way_split(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    sensitive: Union[pd.Series, np.ndarray],
    timestamps: Union[pd.Series, np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> SplitResult:
    """
    Chronological 3-way partition:
    Early -> Train, Mid -> Validation, Late -> Locked Test.
    Strictly verifies max(train_time) <= min(val_time) <= min(test_time).
    """
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)
    s_series = pd.Series(sensitive).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    sort_idx = ts.argsort().to_numpy()
    X_sorted = X_df.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y_series.iloc[sort_idx].reset_index(drop=True)
    s_sorted = s_series.iloc[sort_idx].reset_index(drop=True)
    ts_sorted = ts.iloc[sort_idx].reset_index(drop=True)

    n = len(X_sorted)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, n)

    train_ts_max = ts_sorted.iloc[train_idx].max()
    val_ts_min = ts_sorted.iloc[val_idx].min()
    val_ts_max = ts_sorted.iloc[val_idx].max()
    test_ts_min = ts_sorted.iloc[test_idx].min()

    if train_ts_max > val_ts_min or val_ts_max > test_ts_min:
        raise TemporalLeakageError("Temporal ordering violation detected across splits!")

    return SplitResult(
        X_train=X_sorted.iloc[train_idx].copy(),
        y_train=y_sorted.iloc[train_idx].copy(),
        s_train=s_sorted.iloc[train_idx].copy(),
        X_val=X_sorted.iloc[val_idx].copy(),
        y_val=y_sorted.iloc[val_idx].copy(),
        s_val=s_sorted.iloc[val_idx].copy(),
        X_test=X_sorted.iloc[test_idx].copy(),
        y_test=y_sorted.iloc[test_idx].copy(),
        s_test=s_sorted.iloc[test_idx].copy(),
        regime=SplitRegime.TEMPORAL_HOLDOUT,
    )


def stratified_3way_split(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    sensitive: Union[pd.Series, np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """
    Standard stratified random 3-way split (IID Holdout).
    """
    y_series = pd.Series(y).reset_index(drop=True)
    s_series = pd.Series(sensitive).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    stratify = y_series if y_series.nunique() > 1 else None

    # Step 1: Split off train (e.g. 60%)
    rem_ratio = val_ratio + test_ratio
    X_train, X_rem, y_train, y_rem, s_train, s_rem = train_test_split(
        X_df, y_series, s_series,
        test_size=rem_ratio,
        random_state=seed,
        stratify=stratify,
    )

    # Step 2: Split remainder into val and test (50/50 of remainder)
    rem_stratify = y_rem if y_rem.nunique() > 1 else None
    test_fraction_of_rem = test_ratio / rem_ratio
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_rem, y_rem, s_rem,
        test_size=test_fraction_of_rem,
        random_state=seed,
        stratify=rem_stratify,
    )

    return SplitResult(
        X_train=X_train.copy(),
        y_train=y_train.copy(),
        s_train=s_train.copy(),
        X_val=X_val.copy(),
        y_val=y_val.copy(),
        s_val=s_val.copy(),
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        s_test=s_test.copy(),
        regime=SplitRegime.IID_HOLDOUT,
    )
