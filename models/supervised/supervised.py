from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from time import time
import numpy as np

from active_learning.almodel import ALModel
from datasets.conll2002 import Conll2002Dataset
from datasets.conll2003 import Conll2003Dataset


class SupervisedModel(ALModel, nn.Module):

    def __init__(self, n_classes, hidden_dim, lstm_layers,
                 lstm_dropout=0.1, output_dropout=0.1):
        super().__init__()
        self.n_classes = n_classes
        embedding_dim = 300
        #self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        #self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, dropout=lstm_dropout,
                            bidirectional=True)
        self.output_dropout = nn.Dropout(p=output_dropout)
        self.fc1 = nn.Linear(2 * hidden_dim, n_classes)
        logging.debug("Net:\n%r", self)

    def get_embedded(self, word_indexes):
        return self.embedding(word_indexes)

    def forward(self, packed_sents):
        embedded_sents = packed_sents.data
        embedded_sents_packed = nn.utils.rnn.PackedSequence(embedded_sents,
                                                            packed_sents.batch_sizes)
        out_packed_sequence, _ = self.lstm(embedded_sents_packed)
        out = self.fc1(self.output_dropout(out_packed_sequence.data))
        return F.log_softmax(out, dim=1)

    def predict(self, sents, device):
        x = nn.utils.rnn.pack_sequence(sents)

        if device.type == 'cuda':
            x = x.cuda()

        out = self.forward(x)
        del x
        predictions = torch.argmax(out, dim=1).cpu().numpy()
        out = out.cpu().numpy()

        return out, predictions

    def train_batch(self, sents, sents_tags, device = None, lr=0.01):
        self.train()
        x = nn.utils.rnn.pack_sequence(sents)
        y = nn.utils.rnn.pack_sequence(sents_tags)
        if device.type == 'cuda':
            x, y = x.cuda(), y.cuda()
        out = self.forward(x)
        loss = F.nll_loss(out, y.data)

        loss.backward()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer.step()
        predictions = torch.argmax(out, dim=1).cpu()
        out = out.cpu()
        loss = loss.cpu()
        del x
        torch.cuda.empty_cache()

        return out, loss, predictions

    def train_epoch(self, x, y, batch_size, device, lr=0.01, verbose=False):
        self.train()
        last_log_time = time()
        for batch_idx, (x_batch, y_batch) in enumerate(self._data_generator(x, y, batch_size)):
            self.zero_grad()
            out, loss, predictions = self.train_batch(x_batch, y_batch, device, lr)

            if verbose and ((time() - last_log_time) > 5 or batch_idx == 0):

                last_log_time = time()

                # Calculate accuracy
                n_correct_predictions = 0
                word_count = 0
                packed_y = nn.utils.rnn.pack_sequence(y_batch).data
                for y, prediction in zip(packed_y, predictions):
                    word_count += 1
                    if torch.all(y == prediction):
                        n_correct_predictions += 1

                acc = n_correct_predictions / word_count

                print("Batch %d, number of processed sentences %d, loss %.3f, accuracy %.2f" %
                      (batch_idx + 1, batch_size * (batch_idx + 1), loss.item(), acc))


    def evaluate(self, x, y, batch_size, device):
        self.eval()
        with torch.no_grad():
            word_count = 0
            n_correct_predictions = 0
            for batch_idx, (x_batch, y_batch) in enumerate(self._data_generator(x, y, batch_size)):
                out, predictions = self.predict(x_batch, device)
                packed_y = torch.nn.utils.rnn.pack_sequence(y_batch)
                for y, prediction in zip(packed_y.data, predictions):
                    word_count += 1
                    if torch.all(y == prediction):
                        n_correct_predictions += 1
            return n_correct_predictions / word_count

    def score(self, x, y, batch_size, device) -> Dict[str, float]:
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        self.eval()
        y_gt = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(self._data_generator(x, y, batch_size)):
                out, predictions = self.predict(x_batch, device)
                packed_y = torch.nn.utils.rnn.pack_sequence(y_batch)
                y_gt.append(packed_y.data)
                y_pred.append(predictions)
        y_gt = np.hstack(y_gt)
        y_pred = np.hstack(y_pred)
        return {"acc": accuracy_score(y_gt, y_pred),
                "f1": f1_score(y_gt, y_pred, average='macro'),
                "precision": precision_score(y_gt, y_pred, average='macro'),
                "recall": recall_score(y_gt, y_pred, average='macro')
                }


    def viterbi_paths(self, X: List[np.ndarray], device, **kwargs):
        self.eval()
        with torch.no_grad():
            paths = []
            path_probs = []
            for batch_idx, x_batch in enumerate(self._data_generator(X)):
                sigmoids, predictions = self.predict(x_batch, device)

                paths.append(np.array([predictions]))
                path_probs.append(np.array([sigmoids.max(axis=1).sum()]))
        return paths, path_probs

    def state_confidences(self, X: List[np.ndarray], device, **kwargs) -> List[np.ndarray]:
        self.eval()
        with torch.no_grad():
            state_confidences = []
            for batch_idx, x_batch in enumerate(self._data_generator(X)):
                sigmoids, predictions = self.predict(x_batch, device)
                state_confidences.append(sigmoids)
        return state_confidences

    def _data_generator(self, x, y=None, batch_size=1):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            x_batch.sort(key=lambda l: len(l), reverse=True)
            if y:
                y_batch = y[i:i + batch_size]
                y_batch.sort(key=lambda l: len(l), reverse=True)
                yield [torch.Tensor(s) for s in x_batch], [torch.LongTensor(s) for s in y_batch]
            else:
                yield [torch.Tensor(s) for s in x_batch]


if __name__ == "__main__":
    dataset = Conll2003Dataset(save_file_path='../../datasets/saves/conll2003NER.pkl', task='NER')

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    train_dataset, test_dataset = dataset.train_test_split()
    ltrain_dataset, utrain_dataset = train_dataset.split(0.1)

    model = SupervisedModelAL(dataset.max_tag_idx + 1, 100, 2, 0.8)
    if device.type == 'cuda':
        model = model.cuda()

    batch_size = 200
    for epoch_idx in range(10):
        print("--------- EPOCH: %d -------------" % (epoch_idx + 1))
        model.train_epoch(train_dataset.x_embeddings, train_dataset.y, batch_size, device, 0.01)
        print(f"Labeled train score: {model.score(train_dataset.x_embeddings, train_dataset.y, batch_size, device)}")
        print(f"Unlabeled train score: {model.score(utrain_dataset.x_embeddings, utrain_dataset.y, batch_size, device)}")
        print(f"Test score:  {model.score(test_dataset.x_embeddings, test_dataset.y, batch_size, device)}")

    #torch.save(model, 'models/supervised/saves/supervised_lmpretraining_5epochs.pkl')
