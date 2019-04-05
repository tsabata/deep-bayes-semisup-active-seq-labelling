from typing import List, Dict, Any

import numpy as np

from active_learning.almodel import ALModel
from active_learning.labeler import Labeler


def get_mostinformative_index_sequential(viterbi_scores: List[np.ndarray] = None,
                                         states_confidences: List[np.ndarray] = None,
                                         similarity: np.ndarray = None,
                                         similarity_beta: float = None,
                                         strategy="least_confident",
                                         n=1) -> List[int]:
    """
    Return the most informative sequence

    Parameters
    ----------
    viterbi_scores : np.ndarray
        shape: 1d: sequence_scores (logarithms of probs)
    states_confidences : List[np.ndarray]
        list of arrays(time, token_confidences)
        (logarithms of probs)
    similarity : np.ndarray
        representativness
    strategy : {least_confident, sequence_entropy, token_entropy, random} (default="least_confident)
        informativness measure
    n : int (default=1)
        number of queries in one batch
    Returns
    -------
    list of int
        list of indexes of the most informative sequences
    """
    if strategy == 'least_confident':
        lc = 1 - np.exp(np.array([score[0] for score in viterbi_scores]))
        if similarity_beta > 0:
            lc = lc*(similarity**similarity_beta)
        return np.argsort(lc)[::-1][:n].tolist()
    elif strategy == 'token_entropy':
        # We assume confidences to be logs
        seq_ents = []
        for seq in states_confidences:
            entropy = (-1.0 / seq.shape[0]) * np.sum(np.sum(np.nan_to_num(np.multiply(np.exp(seq), seq)), axis=1), axis=0)
            seq_ents.append(entropy)
        seq_ents = np.array(seq_ents)
        if similarity_beta > 0:
            seq_ents = seq_ents*(similarity**similarity_beta)
        return np.argsort(seq_ents)[::-1][:n].tolist()
    elif strategy == 'margin':
        margin = np.array([np.abs(np.exp(score[0]) - np.exp(score[1])) if len(score) > 1 else 0.0
                      for score in viterbi_scores])
        if similarity_beta > 0:
            margin = margin*(similarity**similarity_beta)
        return np.argsort(margin)[::-1][:n].tolist()
    elif strategy == 'total_token_entropy':
        # We assume confidences to be logs
        seq_ents = []
        for seq in states_confidences:
            entropy = -1.0 * np.sum(np.sum(np.nan_to_num(np.multiply(np.exp(seq), seq)), axis=1), axis=0)
            seq_ents.append(entropy)
        seq_ents = np.array(seq_ents)
        if similarity_beta > 0:
            seq_ents = seq_ents*(similarity**similarity_beta)
        return np.argsort(seq_ents)[::-1][:n].tolist()
    elif strategy == 'nbest_sequence_entropy':
        entropies = np.array([-1 * np.sum(np.multiply(np.exp(x), x)) for x in viterbi_scores])
        if similarity_beta > 0:
            entropies = entropies*(similarity**similarity_beta)
        return np.argsort(entropies)[::-1][:n].tolist()
    elif strategy == 'random':
        if viterbi_scores:
            arr = np.arange(len(viterbi_scores))
        elif states_confidences:
            arr = np.arange(len(states_confidences))
        else:
            raise ValueError("Pass either viterbi_scores or state_confidences")
        np.random.shuffle(arr)
        return arr[:n].tolist()
    else:
        raise NotImplementedError(strategy + " is not implemented.")


def active_learning_step(unlabeled_x: List[np.ndarray], unlabeled_y: List[np.ndarray],
                         id_sim: np.ndarray,
                         id_beta: float,
                         model: ALModel, model_kwargs : Dict[str, Any],
                         labeler: Labeler,
                         strategy="least_confident",
                         sequences: int = 100):
    viterbi_paths, viterbi_scores = model.viterbi_paths(unlabeled_x, **model_kwargs)
    states_confidence = model.state_confidences(unlabeled_x, **model_kwargs)
    indices = get_mostinformative_index_sequential(viterbi_scores, states_confidence,
                                                   id_sim, id_beta, strategy, sequences)
    annotated_cnt_total = 0
    pseudolabeld_ratios_all = []
    threshold_ratios_all = []
    y_labeled_all = []
    for i in indices:
        y_labeled, annotated_cnt, pseudolabeled_ratios, threshold_ratios = labeler.label(
            unlabeled_x[i], confidences=states_confidence[i], y=unlabeled_y[i],
            path=viterbi_paths[i][0])
        annotated_cnt_total += annotated_cnt
        pseudolabeld_ratios_all.append(pseudolabeled_ratios)
        threshold_ratios_all.append(threshold_ratios)
        y_labeled_all.append((unlabeled_x[i], y_labeled))
    return indices, y_labeled_all, annotated_cnt_total, np.array(pseudolabeld_ratios_all), np.array(threshold_ratios_all)


def active_learning_step_limited_tokens(
        unlabeled_x: List[np.ndarray], unlabeled_y: List[np.ndarray],
        id_sim: np.ndarray,
        id_beta: float,
        model: ALModel, model_kwargs : Dict[str, Any],
        labeler: Labeler,
        strategy="least_confident",
        tokens_limit: int = 1000):
    viterbi_paths, viterbi_scores = model.viterbi_paths(unlabeled_x, **model_kwargs)
    states_confidence = model.state_confidences(unlabeled_x, **model_kwargs)
    indices = get_mostinformative_index_sequential(viterbi_scores, states_confidence,
                                                   id_sim, id_beta, strategy, tokens_limit)
    annotated_cnt_total = 0
    pseudolabeld_ratios_all = []
    threshold_ratios_all = []
    y_labeled_all = []
    indices_tmp = []
    loop_i = 0
    while annotated_cnt_total < tokens_limit:
        index = indices[loop_i]
        y_labeled, annotated_cnt, pseudolabeled_ratios, threshold_ratios = labeler.label(
            unlabeled_x[index], confidences=states_confidence[index], y=unlabeled_y[index],
            path=viterbi_paths[index][0])
        annotated_cnt_total += annotated_cnt
        pseudolabeld_ratios_all.append(pseudolabeled_ratios)
        threshold_ratios_all.append(threshold_ratios)
        y_labeled_all.append((unlabeled_x[index], y_labeled))
        loop_i += 1
        indices_tmp.append(index)
    return indices_tmp, y_labeled_all, annotated_cnt_total,\
           np.array(pseudolabeld_ratios_all), np.array(threshold_ratios_all)


def active_learning_loop(labeled_x: List[np.ndarray], labeled_y: List[np.ndarray],
                         unlabeled_x: List[np.ndarray], unlabeled_y: List[np.ndarray],
                         id_sim: List[float],
                         id_beta: float,
                         model, model_kwargs : Dict[str, Any], strategy, labeler,
                         sentences=100):
    indices, annotated, annotated_cnt, pseudolabeld_ratios, threshold_ratios = active_learning_step(
        unlabeled_x, unlabeled_y, np.array(id_sim), id_beta, model, model_kwargs, labeler, strategy, sentences)
    batch = []
    for index, i in enumerate(reversed(sorted(indices))):
        labeled_x.append(annotated[index][0])
        labeled_y.append(annotated[index][1])
        batch.append(annotated[index])
        del unlabeled_x[i], unlabeled_y[i], id_sim[i]
    return indices, annotated_cnt, pseudolabeld_ratios, threshold_ratios, batch


def active_learning_loop_limited_tokens(labeled_x: List[np.ndarray], labeled_y: List[np.ndarray],
                          unlabeled_x: List[np.ndarray], unlabeled_y: List[np.ndarray],
                          id_sim: List[float],
                          id_beta: float,
                          model, model_kwargs : Dict[str, Any], strategy, labeler,
                          al_batch_size=1000):
    indices, annotated, annotated_cnt, pseudolabeld_ratios, threshold_ratios = active_learning_loop_limited_tokens(
        labeled_x, labeled_y,
        unlabeled_x, unlabeled_y, np.array(id_sim), id_beta,
        model, model_kwargs, labeler, strategy, al_batch_size)
    batch = []
    for index, i in enumerate(reversed(sorted(indices))):
        labeled_x.append(annotated[index][0])
        labeled_y.append(annotated[index][1])
        batch.append(annotated[index])
        del unlabeled_x[i], unlabeled_y[i], id_sim[i]
    return indices, annotated_cnt, pseudolabeld_ratios, threshold_ratios, batch
