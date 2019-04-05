import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from time import time

from datasets.conll2002 import Conll2002Dataset


class LanguageModel(nn.Module):

    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, lstm_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_dim, input_vocab_size)
        logging.debug("Net:\n%r", self)

    def get_embedded(self, word_indexes):
        return self.embedding(word_indexes)

    def forward(self, packed_sents):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        embedded_sents = nn.utils.rnn.PackedSequence(self.get_embedded(packed_sents.data),
                                                     packed_sents.batch_sizes)
        out_packed_sequence, _ = self.lstm(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)

    def predict(self, sents, device):
        x = nn.utils.rnn.pack_sequence([sent[:-1] for sent in sents])

        if device.type == 'cuda':
            x = x.cuda()

        out = self.forward(x)
        del x
        predictions = torch.argmax(out, dim=1).cpu()
        out = out.cpu()

        return out, predictions

    def train_batch(self, sents, device, lr=0.01):
        x = nn.utils.rnn.pack_sequence([sent[:-1] for sent in sents])
        y = nn.utils.rnn.pack_sequence([sent[1:] for sent in sents])
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
        y = y.cpu()
        del x
        torch.cuda.empty_cache()

        return out, loss, predictions

    def train_epoch(self, x, batch_size, device, lr=0.01, verbose=False):
        self.train()
        last_log_time = time()
        for batch_idx, x_batch in enumerate(self._data_generator(x, batch_size)):
            self.zero_grad()
            out, loss, _ = self.train_batch(x_batch, device, lr)

            if verbose and ((time() - last_log_time) > 5 or batch_idx == 0):
                last_log_time = time()

                # Calculate perplexity.
                packed_y = torch.nn.utils.rnn.pack_sequence([sent[1:] for sent in x_batch])
                prob = out.exp()[torch.arange(0, packed_y.data.shape[0], dtype=torch.int64),
                                 packed_y.data]
                perplexity = 2 ** prob.log2().neg().mean().item()

                print("Batch %d, number of processed sentences %d, loss %.3f, perplexity %.2f" %
                      (batch_idx + 1, batch_size * (batch_idx + 1), loss.item(), perplexity))

    def evaluate(self, x, batch_size, device):
        self.eval()
        with torch.no_grad():
            for batch_idx, x_batch in enumerate(self._data_generator(x, batch_size)):
                out, _ = self.predict(x_batch, device)
                packed_y = torch.nn.utils.rnn.pack_sequence([sent[1:] for sent in x_batch])

                # Calculate perplexity.
                prob = out.exp()[torch.arange(0, packed_y.data.shape[0], dtype=torch.int64),
                                 packed_y.data]
                perplexity = 2 ** prob.log2().neg().mean().item()
            return perplexity

    def _data_generator(self, x, batch_size):
        # only sentences with more than 1 word
        x = [sent for sent in x if len(sent) > 1]

        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            x_batch.sort(key=lambda l: len(l), reverse=True)
            yield [torch.LongTensor(s) for s in x_batch]


if __name__ == "__main__":
    dataset = Conll2002Dataset(save_file_path='../../datasets/saves/conll2002.pkl')

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    train_dataset, test_dataset = dataset.split(0.8)
    ltrain_dataset, utrain_dataset = train_dataset.split(0.1)

    model = LanguageModel(dataset.max_word_idx + 1, 20, 100, 1, 0)
    if device.type == 'cuda':
        model = model.cuda()

    batch_size = 50
    for epoch_idx in range(5):
        print("--------- EPOCH: %d -------------" % (epoch_idx + 1))
        model.train_epoch(utrain_dataset.x, batch_size, device, 0.01, True)
        print("Labeled train perplexity: %.2f" % model.evaluate(ltrain_dataset.x, batch_size, device))
        print("Unlabeled train perplexity: %.2f" % model.evaluate(utrain_dataset.x, batch_size, device))
        print("Test perplexity: %.2f" % model.evaluate(test_dataset.x, batch_size, device))
        print('')

    torch.save(model, 'saves/lm_5epochs.pkl')
