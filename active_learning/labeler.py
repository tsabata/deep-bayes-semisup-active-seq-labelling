from typing import Tuple, List

import numpy as np

class Labeler:

    def label(self, X, confidences=None, y=None, path=None) -> Tuple[np.ndarray, int, List[float], List[float]]:
        prediction_confidences = np.exp(np.array([confidences[time, tag] for time, tag in enumerate(path)]))
        wrong_prediction_confidences = np.sort((path != y) * prediction_confidences)[::-1]
        if len(np.unique(wrong_prediction_confidences))>1:
            wrong_prediction_confidences[wrong_prediction_confidences == 0] = sorted(np.unique(wrong_prediction_confidences))[1]
        error_thresholds = [wrong_prediction_confidences[
                                min(int(np.ceil((len(wrong_prediction_confidences) / 100) * level)),
                                    len(wrong_prediction_confidences) - 1)] for level in
                            [0, 1, 5, 10, 15]]
        pseudo_annotated = [np.sum(prediction_confidences > th)/len(prediction_confidences) for th in error_thresholds]
        return y, len(path), pseudo_annotated, error_thresholds
