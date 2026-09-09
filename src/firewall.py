# ============================================================================
# Model-Selection Firewall
# Enforces architectural guardrails preventing holdout test set contamination
# during hyperparameter optimization, threshold tuning, and model selection.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


class ModelSelectionFirewallViolation(RuntimeError):
    """Raised when locked holdout test data is improperly passed to tuning or selection routines."""
    pass


class PolicyLockError(RuntimeError):
    """Raised when an attempt is made to evaluate on locked test data before policy freezing."""
    pass


@dataclass
class LockedTestData:
    """
    Immutable container for the final locked holdout test set.
    Forbids direct access to fitting or optimization methods.
    """
    X: pd.DataFrame
    y: pd.Series
    sensitive: pd.Series
    dataset_name: str
    _is_locked: bool = True

    def __post_init__(self):
        # Freeze dataframes to prevent in-place mutation
        self.X.flags.writeable = False

    def fit(self, *args, **kwargs):
        raise ModelSelectionFirewallViolation(
            f"FIREWALL VIOLATION: Cannot call .fit() on LockedTestData for '{self.dataset_name}'."
        )

    def tune(self, *args, **kwargs):
        raise ModelSelectionFirewallViolation(
            f"FIREWALL VIOLATION: Cannot perform tuning on LockedTestData for '{self.dataset_name}'."
        )

    def select(self, *args, **kwargs):
        raise ModelSelectionFirewallViolation(
            f"FIREWALL VIOLATION: Cannot perform model selection on LockedTestData for '{self.dataset_name}'."
        )


@dataclass
class LockedPolicy:
    """
    Represents a frozen, validation-selected model and post-processing policy.
    Can only be evaluated against LockedTestData after formal policy freezing.
    """
    model_name: str
    estimator: Any
    threshold_optimizer: Optional[Any] = None
    validation_score: float = 0.0
    validation_metrics: Optional[Dict[str, float]] = None
    is_frozen: bool = True

    def predict(self, X: np.ndarray, sensitive_features: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_frozen:
            raise PolicyLockError("Cannot predict with an unfrozen policy.")
        if self.threshold_optimizer is not None:
            return np.asarray(self.threshold_optimizer.predict(X, sensitive_features=sensitive_features)).astype(int)
        return np.asarray(self.estimator.predict(X)).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_frozen:
            raise PolicyLockError("Cannot predict with an unfrozen policy.")
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] >= 2:
                return np.asarray(proba)[:, 1]
            return np.asarray(proba).ravel()
        return None


def verify_no_test_leakage_in_args(*args, **kwargs) -> None:
    """
    Inspects positional and keyword arguments to guarantee no LockedTestData
    is accidentally passed to tuning, HPO, or selection functions.
    """
    for idx, arg in enumerate(args):
        if isinstance(arg, LockedTestData):
            raise ModelSelectionFirewallViolation(
                f"FIREWALL VIOLATION: Argument at position {idx} is LockedTestData. "
                "Locked test data must NEVER be passed to training, tuning, or model selection functions."
            )
    for key, val in kwargs.items():
        if isinstance(val, LockedTestData):
            raise ModelSelectionFirewallViolation(
                f"FIREWALL VIOLATION: Keyword argument '{key}' is LockedTestData. "
                "Locked test data must NEVER be passed to training, tuning, or model selection functions."
            )


def firewall_protected(func: Callable) -> Callable:
    """
    Decorator enforcing that the wrapped function cannot receive LockedTestData.
    """
    def wrapper(*args, **kwargs):
        verify_no_test_leakage_in_args(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper
