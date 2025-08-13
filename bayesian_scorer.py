import logging
from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.naive_bayes import GaussianNB  # type: ignore
    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    HAS_SKLEARN = False

from .pattern_database import PatternDatabase, PatternRecord

logger = logging.getLogger(__name__)


class BayesianPatternScorer:
    """Trains per-pattern-type Gaussian Naive Bayes models from confirmed history
    and predicts success probability for new patterns.
    """

    def __init__(self) -> None:
        self.db = PatternDatabase()
        # pattern_type -> (model, feature_keys)
        self.models: Dict[str, Tuple[GaussianNB, list[str]]] = {}

    def _prepare_dataset(self, pattern_type: str):
        historical = self.db.get_historical(pattern_type)
        if len(historical) < 20:
            return None, None
        # Establish a stable feature schema from first record
        feature_keys = sorted(historical[0].features.keys())
        X = []
        y = []
        for rec in historical:
            try:
                X.append([float(rec.features.get(k, 0.0)) for k in feature_keys])
                y.append(1 if rec.outcome else 0)
            except Exception:
                continue
        if not X or len(set(y)) < 2:
            return None, None
        return np.array(X, dtype=float), np.array(y, dtype=int), feature_keys

    def train(self, pattern_type: str) -> bool:
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available; BayesianPatternScorer disabled")
            return False
        res = self._prepare_dataset(pattern_type)
        if res is None:
            return False
        X, y, feature_keys = res
        try:
            model = GaussianNB()
            model.fit(X, y)
            self.models[pattern_type] = (model, feature_keys)
            return True
        except Exception as exc:
            logger.error(f"Pattern model training failed for {pattern_type}: {exc}")
            return False

    def predict_probability(self, pattern_type: str, features: Dict[str, float]) -> float:
        if not HAS_SKLEARN:
            return 0.5
        if pattern_type not in self.models:
            ok = self.train(pattern_type)
            if not ok:
                return 0.5
        model, feature_keys = self.models[pattern_type]
        try:
            x = np.array([[float(features.get(k, 0.0)) for k in feature_keys]], dtype=float)
            proba = model.predict_proba(x)[0]
            # Return probability of success (class 1)
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as exc:
            logger.warning(f"Pattern probability prediction failed for {pattern_type}: {exc}")
            return 0.5


