from __future__ import annotations

import numpy as np
import numpy.typing as npt


def precision_recall_f1(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
) -> dict[str, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}
