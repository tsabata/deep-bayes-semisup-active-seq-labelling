from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


class ActiveLearningTelemetry:

    def __init__(self, name):
        """
        Contains tuple (annotated_points_cnt, pseudolabeled_point, queried_point, scores: dict)
        """
        self.history = []
        self.name = name

    def add_point(self, annotated_cnt: int, pseudolabeled_ratios: np.ndarray, threshold_ratios: np.ndarray,
                  query: Tuple[np.ndarray, np.ndarray], scores: List[Dict[str, float]]):
        if isinstance(scores, list):
            avg_score = {}
            for k in scores[0].keys():
                avg = 0
                for score in scores:
                    avg += score[k]
                avg = float(avg)/len(scores)
                avg_score[k] = avg
            scores = avg_score
        pseudolabeled_ratios.mean(axis=1)
        self.history.append((annotated_cnt, (pseudolabeled_ratios.mean(axis=0), pseudolabeled_ratios.std(axis=0)),
                             (threshold_ratios.mean(axis=0), threshold_ratios.std(axis=0)), query, scores))

    def annotated_points_cnt(self) -> List[int]:
        return [x[0] for x in self.history]

    def pseudolabeled_ratios_mean(self) -> List[int]:
        return [x[1][0] for x in self.history]

    def pseudolabeled_ratios_std(self) -> List[int]:
        return [x[1][1] for x in self.history]

    def threshold_ratios_mean(self) -> List[int]:
        return [x[2][0] for x in self.history]

    def threshold_ratios_std(self) -> List[int]:
        return [x[2][1] for x in self.history]


    def queries(self) -> List[np.array]:
        return [x[3] for x in self.history]

    def scores(self) -> Dict[str, List[float]]:
        LD = [x[4] for x in self.history]
        return {k: [dic[k] for dic in LD] for k in LD[0]} if LD else {}

    def to_df(self):
        df = pd.DataFrame(self.scores())
        df['manually_annotated'] = self.annotated_points_cnt()
        for i, err in enumerate([0, 1, 5, 10, 15]):
            df[f"pseudolabeled_ratio_mean_{err}"] = np.array(self.pseudolabeled_ratios_mean())[:, i]
            df[f"pseudolabeled_ratio_std_{err}"] = np.array(self.pseudolabeled_ratios_std())[:, i]
            df[f"threshold_ratio_mean_{err}"] = np.array(self.threshold_ratios_mean())[:, i]
            df[f"threshold_ratio_std_{err}"] = np.array(self.threshold_ratios_std())[:, i]
        df['strategy'] = self.name
        return df

    def __repr__(self):
        return f"Strategy: {self.name}, \n" \
            f"Annotated: {self.annotated_points_cnt()}, \n" \
            f"Pseudolabeled_ratios_mean: {self.pseudolabeled_ratios_mean()}, \n" \
            f"Threshold_ratios_mean: {self.threshold_ratios_mean()}, \n" \
            f"Scores: {self.scores()}"
