import abc
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ALModel(abc.ABC):

    @abc.abstractmethod
    def viterbi_paths(self, X: List[np.ndarray], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes Viterbi paths and their scores.

        Parameters
        ----------
        X : list
            list of sequences represented as np.array
        Returns
        -------
        paths : list
            list of best paths represented as np.array

        path_scores : list
            list of path scores
        """

    @abc.abstractmethod
    def state_confidences(self, X: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Finds marginal probabilities of tags for each member of sequence.

        Parameters
        ----------
        X : list
            list of sequences represented as np.array

        Returns
        -------
        list
            list of numpy arrays representing tags confidences
            array-like (sequence_len, possible_tokens_cnt)
        """

    def score(self, X: List[np.ndarray], y: List[np.ndarray], **kwargs) -> Dict[str, float]:
        y_pred_flat = np.concatenate(self.predict(X)).ravel()
        y_flat = np.concatenate(y).ravel()
        return {"acc": accuracy_score(y_flat, y_pred_flat),
                "f1_": f1_score(y_flat, y_pred_flat, average='macro'),
                "precision": precision_score(y_flat, y_pred_flat, average='macro'),
                "recall": recall_score(y_flat, y_pred_flat, average='macro')
                }

    @abc.abstractmethod
    def predict(self, X: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Predicts labels for each sequence in X

        Parameters
        ----------
        X : list
            list of sequences represented as numpy arrays.

        Returns
        -------
        list
            list of predictions for each sequence
        """

    @abc.abstractmethod
    def train_batch(self, sents, sents_tags, device = None, lr=0.01):
        pass
