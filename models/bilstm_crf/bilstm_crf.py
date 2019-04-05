from typing import List, Optional, Dict, Union, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from active_learning.almodel import ALModel
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets.conll2002 import Conll2002Dataset
from datasets.conll2003 import Conll2003Dataset


def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.
    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.
        :param feats: output of word RNN/BLSTM, a tensor of dim. (batch_size, timesteps, hidden_dim)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores_b = self.emission(feats)  # (batch_size, timesteps, tagset_size)

        # (batch_size, timesteps, tagset_size, tagset_size)
        emission_scores = emission_scores_b.unsqueeze(2).expand(self.batch_size, self.timesteps,
                                                                self.tagset_size, self.tagset_size)
        transition = self.transition.unsqueeze(0).unsqueeze(0).expand(
            self.batch_size, self.timesteps, self.tagset_size, self.tagset_size)

        return emission_scores, transition


class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, n_labels, start_label, end_label):
        super(ViterbiLoss, self).__init__()
        self.n_labels = n_labels
        self.start_label = start_label
        self.end_label = end_label

    def forward(self, emission_scores, transition, targets, lengths):
        """
        Forward propagation.
        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: word sequence lengths
        :return: viterbi loss
        """

        scores = emission_scores + transition

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Gold score

        targets = targets.unsqueeze(2)

        # (batch_size, word_pad_len)
        scores_at_targets = \
            torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(2)

        # Everything is already sorted by lengths
        scores_at_targets, _ = nn.utils.rnn.pack_padded_sequence(scores_at_targets, lengths,
                                                                 batch_first=True)
        gold_score = scores_at_targets.sum()

        # All paths' scores

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.n_labels, device=scores.device)

        for t in range(max(lengths)):
            # effective batch size (sans pads) at this timestep
            batch_size_t = sum([l > t for l in lengths])
            if t == 0:
                # (batch_size, tagset_size)
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_label, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep,
                # and log-sum-exp. Remember, the cur_tag of the previous timestep is the prev_tag
                # of this timestep. So, broadcast prev. timestep's cur_tag scores along current
                # timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_label].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        del scores_upto_t
        del scores_at_targets

        return viterbi_loss


class ViterbiDecoderForward(nn.Module):
    """
    Viterbi Decoder.
    """

    def __init__(self, n_labels, start_label, end_label):
        super(ViterbiDecoderForward, self).__init__()
        self.n_labels = n_labels
        self.start_label = start_label
        self.end_label = end_label

    def forward(self, emission_scores, transition, lengths):

        scores = emission_scores + transition
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.n_labels, device=scores.device)

        best_scores = torch.zeros(batch_size, self.n_labels, device=scores.device)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current
        # tag Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, max(lengths), self.n_labels), dtype=torch.long,
                                  device=scores.device) * self.end_label

        for t in range(max(lengths)):
            batch_size_t = sum(
                [l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                # (batch_size, tagset_size)
                best_scores[:batch_size_t] = scores[:batch_size_t, t, self.start_label]
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.n_labels),
                                                               dtype=torch.long) * self.start_label
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_label, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep,
                # and choose the previous timestep that corresponds to the max. accumulated score
                # for each current timestep
                best_scores[:batch_size_t], backpointers[:batch_size_t, t, :] = torch.max(
                    scores[:batch_size_t, t] + best_scores[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long,
                              device=scores.device)
        # the pointers at the ends are all <end> tags
        pointer = torch.ones((batch_size, 1), dtype=torch.long,
                             device=scores.device) * self.end_label

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long,
                                                     device=scores.device) * self.start_label)

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if
        # any)
        decoded = torch.cat([decoded[:, 1:], torch.ones((batch_size, 1), dtype=torch.long,
                                                        device=scores.device) * self.end_label],
                            dim=1)

        # all_path_scores = scores_upto_t[:, self.end_label]
        # best_path_scores = best_scores[:, self.end_label]

        # targets = decoded.unsqueeze(2)
        # 
        # # (batch_size, word_pad_len)
        # scores_at_targets = \
        #     torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(2)
        # 
        # best_path_scores = torch.zeros_like(all_path_scores)
        # for seq_idx, seq_length in enumerate(lengths):
        #     best_path_scores[seq_idx] = scores_at_targets[seq_idx, :seq_length].sum()

        scores = -(scores_upto_t[:, self.end_label] - best_scores[:, self.end_label])

        del backpointers
        del best_scores
        del pointer
        del scores_upto_t
        torch.cuda.empty_cache()

        return decoded, scores


class BilstmCRF(nn.Module, ALModel):

    def __init__(self, input_vocab_size, n_labels, embedding_dim, hidden_dim, lstm_layers,
                 embedding_dropout, lstm_dropout, output_dropout, start_label, end_label,
                 add_embedding_layer=True, embedded_end_word=None, use_fc=False):
        super().__init__()

        self.n_labels = n_labels
        self.start_label = start_label
        self.end_label = end_label

        if add_embedding_layer:
            self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        else:
            # if embedding_dropout > 0:
            #   print('Embedding dropout is not working with custom embeddings.')
            self.embedding = None
            if embedded_end_word is None:
                raise ValueError('Embedded end word is required.')
            self.embedded_end_word = embedded_end_word

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, dropout=lstm_dropout,
                            bidirectional=True)
        self.output_dropout = nn.Dropout(p=output_dropout)

        self.use_fc = use_fc
        if use_fc:
            self.fc1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self.crf = CRF(2 * hidden_dim, n_labels)

        self.loss = ViterbiLoss(n_labels, start_label, end_label)
        self.viterbi_decoder = ViterbiDecoderForward(n_labels, start_label, end_label)

        logging.debug("Net:\n%r", self)

    def get_embedded(self, word_indexes: torch.Tensor) -> torch.Tensor:
        return self.embedding(word_indexes)

    def forward(self, packed_sents):
        if self.embedding is not None:
            embedded = self.embedding_dropout(self.embedding(packed_sents.data))
        else:
            embedded = self.embedding_dropout(packed_sents.data)

        embedded_sents = nn.utils.rnn.PackedSequence(embedded, packed_sents.batch_sizes)

        out_packed_sequence, _ = self.lstm(embedded_sents)

        if self.use_fc:
            out_packed_sequence_data = F.relu(self.fc1(out_packed_sequence.data))
            out_packed_sequence = nn.utils.rnn.PackedSequence(out_packed_sequence_data,
                                                              packed_sents.batch_sizes)

        padded_out, _ = nn.utils.rnn.pad_packed_sequence(out_packed_sequence, batch_first=True)
        padded_out = self.output_dropout(padded_out)

        emission_scores, transition = self.crf(padded_out)

        return emission_scores, transition

    def compute_marginals(self, emission_scores, transition, lengths,
                          experimental_variant: bool = True):

        forward_scores = self.forward_propagation(emission_scores, transition, lengths)
        backward_scores = self.backward_propagation(emission_scores, transition, lengths)

        if experimental_variant:
            transposed_transition = transition.transpose(2, 3)
            transition_comb = transition + transposed_transition

            marginals = torch.zeros_like(forward_scores)
            for t in range(marginals.shape[1] - 1):
                batch_size_t = sum(
                    [l > t for l in lengths])  # effective batch size (sans pads) at this timestep
                if t == 0:
                    marginals[:batch_size_t, t] = \
                        emission_scores[:batch_size_t, t, 0, :] + \
                        log_sum_exp(
                            transposed_transition[:batch_size_t, t, :, :] +
                            backward_scores[:batch_size_t, t + 1].unsqueeze(2),
                            dim=1)
                marginals[:batch_size_t, t] = \
                    emission_scores[:batch_size_t, t, 0, :] + \
                    log_sum_exp(
                        transition_comb[:batch_size_t, t, :, :] +
                        forward_scores[:batch_size_t, t - 1].unsqueeze(2) +
                        backward_scores[:batch_size_t, t + 1].unsqueeze(2),
                        dim=1)
            del transposed_transition
            del transition_comb
            torch.cuda.empty_cache()
        else:
            marginals = forward_scores + backward_scores

        marginals = F.softmax(marginals[:, :-1, :], dim=2)

        return marginals

    def forward_propagation(self, emission_scores, transition, lengths):
        """
        Forward propagation.
        """

        batch_size = emission_scores.shape[0]
        word_pad_len = emission_scores.shape[1]

        scores = emission_scores + transition

        # Create a tensor to hold accumulated sequence scores at each current tag
        forward_scores = torch.zeros(batch_size, word_pad_len, self.n_labels,
                                     device=emission_scores.device)

        for t in range(1, max(lengths)):
            batch_size_t = sum(
                [l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                forward_scores[:batch_size_t, t] = scores[:batch_size_t, t, self.start_label, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep,
                # and log-sum-exp. Remember, the cur_tag of the previous timestep is the prev_tag
                # of this timestep. So, broadcast prev. timestep's cur_tag scores along current
                # timestep's cur_tag dimension.
                forward_scores[:batch_size_t, t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] +
                    forward_scores[:batch_size_t, t - 1].unsqueeze(2),
                    dim=1)

        del scores
        torch.cuda.empty_cache()

        return forward_scores

    def backward_propagation(self, emission_scores, transition, lengths):
        """
        Backward propagation.
        """

        batch_size = emission_scores.shape[0]
        word_pad_len = emission_scores.shape[1]

        scores = emission_scores + transition.transpose(2, 3)

        # Create a tensor to hold accumulated sequence scores at each current tag
        backward_scores = torch.zeros(batch_size, word_pad_len, self.n_labels,
                                      device=emission_scores.device)

        previous_batch_size = 0
        for t in range(max(lengths) - 2, -1, -1):
            # effective batch size (sans pads) at this timestep
            batch_size_t = sum([l > t for l in lengths])

            if previous_batch_size < batch_size_t:
                backward_scores[previous_batch_size:batch_size_t, t] = \
                    scores[previous_batch_size:batch_size_t, t, self.end_label, :]
            if previous_batch_size > 0:
                backward_scores[:previous_batch_size, t] = log_sum_exp(
                    scores[:previous_batch_size, t, :, :] + backward_scores[:previous_batch_size,
                                                            t + 1].unsqueeze(2),
                    dim=1)
            previous_batch_size = batch_size_t

        del scores
        torch.cuda.empty_cache()

        return backward_scores

    def state_confidences(self, x: List[np.ndarray], batch_size=None, device=None, **kwargs) \
            -> List[np.ndarray]:

        with torch.no_grad():

            if batch_size is None:
                batch_size = len(x)
            if device is None:
                device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

            sentence_marginals = []

            lengths_all = [len(s) for s in x]
            old_order_indices = sorted(range(len(x)), key=lengths_all.__getitem__, reverse=True)
            x = x.copy()
            x.sort(key=lambda l: len(l), reverse=True)

            for x_batch in self._data_generator(x, batch_size=batch_size):
                x_batch = self._preprocess(x_batch)
                lengths = [len(sent) for sent in x_batch]

                x_batch = nn.utils.rnn.pack_sequence([torch.Tensor(s) if self.embedding is None else
                                                      torch.LongTensor(s) for s in x_batch])
                if device.type == 'cuda':
                    x_batch = x_batch.cuda()

                emission_scores, transition = self(x_batch)
                del x_batch
                marginals = self.compute_marginals(emission_scores, transition, lengths)
                marginals = marginals.cpu()
                for i, length in enumerate(lengths):
                    sentence_marginals.append(np.log(marginals[i, :length - 1].numpy()))

                del transition
                del marginals
                del emission_scores
                torch.cuda.empty_cache()

            new_order_indicces = sorted(range(len(x)), key=old_order_indices.__getitem__)
            sentence_marginals_old_order = []
            for ind in new_order_indicces:
                sentence_marginals_old_order.append(sentence_marginals[ind])

            return sentence_marginals_old_order

    def predict(self, x: List[np.ndarray], batch_size: Optional[int] = None, device=None) \
            -> List[np.ndarray]:
        """
        Predicts the labeling.

        Parameters
        ----------
        x : List[np.ndarray]
            list of sequences represented as np.array, expected to be sorted by length if using 
            batch_size greater than 1
        batch_size: int, optional (default = None)
            batch size, if set to None than uses batch sizes of len(x)
        """
        with torch.no_grad():

            if batch_size is None:
                batch_size = len(x)

            if device is None:
                device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

            sentence_predictions = []
            for x_batch in self._data_generator(x, batch_size=batch_size):
                x_batch = self._preprocess(x_batch)
                lengths = [len(sent) for sent in x_batch]
                x_batch = nn.utils.rnn.pack_sequence(x_batch)

                if device.type == 'cuda':
                    x_batch = x_batch.cuda()

                emission_scores, transition = self(x_batch)
                del x_batch

                predictions, _ = self.viterbi_decoder(emission_scores, transition, lengths)

                predictions = predictions.cpu()
                del emission_scores
                del transition
                torch.cuda.empty_cache()

                for i, length in enumerate(lengths):
                    # s = predictions[i, :length - 1].numpy()
                    # s = np.ones_like(s) * 9
                    sentence_predictions.append(predictions[i, :length - 1].numpy())
            return sentence_predictions

    def train_batch(self, x, y, device=None, lr=0.01):
        if not isinstance(x[0], torch.Tensor):
            x = [torch.Tensor(s) for s in x]
            y = [torch.Tensor(s) for s in y]
        x, y = self._preprocess(x, y)
        lengths = [len(sent) for sent in x]
        x = nn.utils.rnn.pack_sequence(x)
        y = nn.utils.rnn.pack_sequence(y)

        if device is None:
            device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

        if device.type == 'cuda':
            x, y = x.cuda(), y.cuda()

        emission_scores, transition = self.forward(x)

        padded_sents_tags, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        loss = self.loss(emission_scores, transition, padded_sents_tags, lengths)
        loss.backward()

        optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer.step()
        predictions, _ = self.viterbi_decoder(emission_scores, transition, lengths)

        predictions = predictions.cpu()
        loss = loss.cpu()
        del x
        del y
        torch.cuda.empty_cache()

        return loss, predictions

    def train_epoch(self, x, y, batch_size, device, lr=0.01):
        self.train()
        for batch_idx, (x_batch, y_batch) in enumerate(self._data_generator(x, y, batch_size)):
            self.zero_grad()
            self.train_batch(x_batch, y_batch, device, lr)

    def score(self, x, y, batch_size=None, device=None) -> Dict[str, float]:
        self.eval()

        with torch.no_grad():
            x = x.copy()
            y = y.copy()
            x.sort(key=lambda l: len(l), reverse=True)
            y.sort(key=lambda l: len(l), reverse=True)

            predictions = self.predict(x, batch_size=batch_size, device=device)
            predictions = [torch.LongTensor(p) for p in predictions]

            y = [torch.LongTensor(p) for p in y]

            y_gt = np.hstack(y)
            y_pred = np.hstack(predictions)
            return {"acc": accuracy_score(y_gt, y_pred),
                    "f1": f1_score(y_gt, y_pred, average='macro'),
                    "precision": precision_score(y_gt, y_pred, average='macro'),
                    "recall": recall_score(y_gt, y_pred, average='macro')
                    }

    def viterbi_paths(self, x: List[np.ndarray], batch_size=None, device=None) -> \
            Tuple[List[np.ndarray], List[float]]:
        """
        Input senteces are expected to be sorted by length!
        """

        with torch.no_grad():

            if batch_size is None:
                batch_size = len(x)

            if device is None:
                device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

            lengths_all = [len(s) for s in x]
            old_order_indices = sorted(range(len(x)), key=lengths_all.__getitem__, reverse=True)
            x = x.copy()
            x.sort(key=lambda l: len(l), reverse=True)

            sentence_predictions = []
            path_scores = []
            for x_batch in self._data_generator(x, batch_size=batch_size):

                if not isinstance(x_batch[0], torch.Tensor):
                    x_batch = [torch.LongTensor(s) for s in x_batch]
                x_batch = self._preprocess(x_batch)
                lengths = [len(sent) for sent in x_batch]
                x_batch = nn.utils.rnn.pack_sequence(x_batch)

                if device.type == 'cuda':
                    x_batch = x_batch.cuda()

                emission_scores, transition = self(x_batch)
                del x_batch

                batch_predictions, batch_path_scores = \
                    self.viterbi_decoder(emission_scores, transition, lengths)

                batch_predictions = batch_predictions.cpu()
                del emission_scores
                del transition
                torch.cuda.empty_cache()

                for i, length in enumerate(lengths):
                    sentence_predictions.append(batch_predictions[i, :length - 1].unsqueeze(0)
                                                .numpy())
                path_scores.extend(batch_path_scores.tolist())

            path_scores = np.array([[score] for score in path_scores])

            new_order_indicces = sorted(range(len(x)), key=old_order_indices.__getitem__)
            sentence_predictions_old_order = []
            path_scores_old_order = []
            for ind in new_order_indicces:
                sentence_predictions_old_order.append(sentence_predictions[ind])
                path_scores_old_order.append(path_scores[ind])

            return sentence_predictions_old_order, path_scores_old_order

    def _preprocess(self, x, y: Optional[List[List[int]]] = None):
        x_preprocessed = x[:]

        if y is not None:
            y_preprocessed = y[:]
            y_preprocessed = [torch.cat((s, torch.LongTensor([self.end_label])))
                              for s in y_preprocessed]

        if self.embedding is not None:
            x_preprocessed = [torch.cat((s, torch.LongTensor([self.end_label])))
                              for s in x_preprocessed]
        else:
            x_preprocessed = [
                torch.cat((s, torch.Tensor([self.end_label]) if len(s.shape) == 1 else
                torch.Tensor(self.embedded_end_word).unsqueeze(0)))
                for s in x_preprocessed]
        return (x_preprocessed, y_preprocessed) if y is not None else x_preprocessed

    def _data_generator(self, x, y=None, batch_size=1):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            x_batch.sort(key=lambda l: len(l), reverse=True)
            if y:
                y_batch = y[i:i + batch_size]
                y_batch.sort(key=lambda l: len(l), reverse=True)
                yield [torch.Tensor(s) if self.embedding is None else torch.LongTensor(s)
                       for s in x_batch], \
                      [torch.LongTensor(s) for s in y_batch]
            else:
                yield [torch.Tensor(s) if self.embedding is None else torch.LongTensor(s)
                       for s in x_batch]


if __name__ == "__main__":

    dataset = Conll2003Dataset(save_file_path='../../datasets/saves/conll2003NER.pkl', task='NER')

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    train_dataset, test_dataset = dataset.train_test_split()
    ltrain_dataset, utrain_dataset = train_dataset.split(0.1)

    model = BilstmCRF(dataset.max_word_idx + 1, dataset.max_tag_idx + 1, 300, 100, 1, 0.5, 0.5, 0.5,
                      dataset.tag2idx(dataset.START_WORD), dataset.tag2idx(dataset.END_WORD),
                      False, dataset.word2vec(dataset.END_WORD))

    if device.type == 'cuda':
        model = model.cuda()

    batch_size = 200
    for epoch_idx in range(5):
        print("--------- EPOCH: %d -------------" % (epoch_idx + 1))
        model.train_epoch(ltrain_dataset.x_embeddings, ltrain_dataset.y, batch_size, device, 0.01)
        print(f"Labeled train score: {model.score(train_dataset.x_embeddings, train_dataset.y, batch_size, device)}")
        print(f"Unlabeled train score: {model.score(utrain_dataset.x_embeddings, utrain_dataset.y, batch_size, device)}")
        print(f"Test score:  {model.score(test_dataset.x_embeddings, test_dataset.y, batch_size, device)}")

    os.makedirs('saves', exist_ok=True)
    torch.save(model, 'saves/bilstm_crf.pkl')
