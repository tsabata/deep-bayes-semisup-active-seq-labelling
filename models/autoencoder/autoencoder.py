import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time

from datasets.conll2002 import Conll2002Dataset


class Autoencoder(nn.Module):

    def __init__(self, input_vocab_size, embedding_dim, enc_hidden_dim, enc_lstm_layers,
                 enc_dropout,
                 dec_hidden_dim, dec_lstm_layers, dec_dropout):
        super().__init__()

        # encoder
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.enc_lstm = nn.LSTM(embedding_dim, enc_hidden_dim, enc_lstm_layers, dropout=enc_dropout,
                                bidirectional=True)

        # decoder
        self.dec_lstm = nn.LSTM(enc_hidden_dim, dec_hidden_dim, dec_lstm_layers,
                                dropout=dec_dropout)
        self.dec_fc = nn.Linear(dec_hidden_dim, input_vocab_size)

    def get_embedded(self, word_indexes):
        return self.embedding(word_indexes)

    def forward(self, packed_sents):
        embedded_sents = nn.utils.rnn.PackedSequence(self.get_embedded(packed_sents.data),
                                                     packed_sents.batch_sizes)
        _, (h_n, _) = self.enc_lstm(embedded_sents)

        sent_lengths = nn.utils.rnn.pad_packed_sequence(packed_sents)[1]
        dec_input_sents = []
        for sent_idx in range(len(sent_lengths)):
            sent_length = sent_lengths[sent_idx]
            sent_encoded_vector = h_n[0, sent_idx]
            dec_input_sents.append(sent_encoded_vector.repeat(sent_length, 1))
        dec_packed_input = nn.utils.rnn.pack_sequence(dec_input_sents)

        out_packed_sequence, _ = self.dec_lstm(dec_packed_input)
        out = self.dec_fc(out_packed_sequence.data)

        return F.log_softmax(out, dim=1)

    def predict(self, sents, device):
        x = nn.utils.rnn.pack_sequence(sents)

        if device.type == 'cuda':
            x = x.cuda()

        out = self.forward(x)
        del x
        predictions = torch.argmax(out, dim=1).cpu()
        out = out.cpu()

        return out, predictions

    def train_batch(self, sents, device, lr=0.01):
        x = nn.utils.rnn.pack_sequence(sents)
        y = nn.utils.rnn.pack_sequence(sents)
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
                packed_y = torch.nn.utils.rnn.pack_sequence(x_batch)
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
                packed_y = nn.utils.rnn.pack_sequence(x_batch)

                # Calculate perplexity.
                prob = out.exp()[torch.arange(0, packed_y.data.shape[0], dtype=torch.int64),
                                 packed_y.data]
                perplexity = 2 ** prob.log2().neg().mean().item()
            return perplexity

    def _data_generator(self, x, batch_size):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            x_batch.sort(key=lambda l: len(l), reverse=True)
            yield [torch.LongTensor(s) for s in x_batch]


if __name__ == "__main__":
    dataset = Conll2002Dataset(save_file_path='../../datasets/saves/conll2002.pkl')

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    train_dataset, test_dataset = dataset.split(0.8)
    ltrain_dataset, utrain_dataset = train_dataset.split(0.1)

    model = Autoencoder(dataset.max_word_idx + 1, 20, 100, 1, 0, 100, 1, 0)
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

    torch.save(model, 'saves/autoencoder_5epochs.pkl')
