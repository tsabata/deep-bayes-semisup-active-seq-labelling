from typing import Tuple, Optional, List
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from pathlib import Path

from scipy.spatial.distance import pdist, squareform

import re


class Conll2003Dataset:
    PADDING = 'PAD'
    UNKNOWN_WORD = 'UNK'
    START_WORD = 'START'
    END_WORD = 'END'

    def __init__(self,
                 load: bool = True,
                 save_file_path: Optional[str] = None, task='NER',
                 include_tag_starts: bool = False, include_tag_ends: bool = False,
                 include_word_starts: bool = False, include_word_ends: bool = False):

        save_file_path = save_file_path \
            if save_file_path else f'./datasets/saves/conll2003_{task}.pkl'
        if load:
            if save_file_path is not None and os.path.exists(save_file_path):
                print(f"Loading dataset from {save_file_path}")
                with open(save_file_path, 'rb') as file:
                    dataset = pkl.load(file)
                self.__dict__ = dataset.__dict__
            else:
                dir = os.path.join(Path(__file__).resolve().parent, 'CoNLL-2003')
                glove_dir = os.path.join(Path(__file__).resolve().parent, 'glove')
                train_file = os.path.join(dir, 'eng.train')
                glove_file = os.path.join(glove_dir, 'glove.840B.300d.txt')
                test_file = os.path.join(dir, 'eng.testa')
                sentences, pos_tags, chunk_tags, ner_tags = [], [], [], []
                for dataset_type in ['train', 'test']:
                    if dataset_type == 'train':
                        data_file = train_file
                    else:
                        data_file = test_file

                    with open(data_file, 'r') as f:
                        content = [sentence for sentence in f.read().split("\n\n") if
                                   not sentence.startswith('-DOCSTART-')]
                        words_tuples = [[word.split(" ") for word in sent.split("\n")] for sent in
                                        content]
                        words_tuples = [list(map(list, zip(*sent))) for sent in words_tuples]
                    [sentences.append(sent[0]) for sent in words_tuples if len(sent) == 4]
                    [pos_tags.append(sent[1]) for sent in words_tuples if len(sent) == 4]
                    [chunk_tags.append(sent[2]) for sent in words_tuples if len(sent) == 4]
                    [ner_tags.append(sent[3]) for sent in words_tuples if len(sent) == 4]
                    if dataset_type == "train":
                        self.__split_index = len(sentences)

                if task == 'NER':
                    sentence_tags = ner_tags
                elif task == 'CHUNK':
                    sentence_tags = chunk_tags
                elif task == 'POS':
                    sentence_tags = pos_tags
                else:
                    raise ValueError(f"Unknown task {task}")

                self._words = set([])
                self._tags = set([])

                for sentence in sentences:
                    for word in sentence:
                        self._words.add(word)
                for sentence_tag in sentence_tags:
                    for tag in sentence_tag:
                        self._tags.add(tag)

                self._words = list(self._words)
                self._tags = list(self._tags)

                self._words.insert(0, Conll2003Dataset.UNKNOWN_WORD)
                self._words.insert(0, Conll2003Dataset.END_WORD)
                self._words.insert(0, Conll2003Dataset.START_WORD)
                self._words.insert(0, Conll2003Dataset.PADDING)

                self._tags.insert(0, Conll2003Dataset.UNKNOWN_WORD)
                self._tags.insert(0, Conll2003Dataset.END_WORD)
                self._tags.insert(0, Conll2003Dataset.START_WORD)
                self._tags.insert(0, Conll2003Dataset.PADDING)

                self._max_word_idx = len(self._words)
                self._max_tag_idx = len(self._tags)

                self._word2idx = {word: idx for idx, word in enumerate(self._words)}
                self._word2vec = {}
                self._tag2idx = {tag: idx for idx, tag in enumerate(self._tags)}

                if os.path.exists(glove_file):
                    with open(glove_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.split(' ')[0]
                            if word in self._word2idx.keys():
                                self._word2vec[word] = np.array([float(x) for x in line.split(' ')[1:]])
                    uncategorised = set(self._word2idx.keys()) - set(self._word2vec.keys())
                    for word in uncategorised:
                        if re.search(r'\d', word):
                            self._word2vec[word] = self._word2vec['0']
                        else:
                            self._word2vec[word] = self._word2vec['UNK']

                    self._sentences_embeddings = [[self._word2vec[word] for word in sentence]
                                                  for sentence in sentences]

                else:
                    print('No glove embeddings loaded.')

                # Convert input words to integers:
                self._isentences = [[self._word2idx[word] for word in sentence]
                                    for sentence in sentences]

                self._sentence_itags = [[self._tag2idx[word] for word in sentence]
                                        for sentence in sentence_tags]

                if include_word_starts:
                    for sentence in self._isentences:
                        sentence.insert(0, self.word2idx(Conll2003Dataset.START_WORD))

                if include_word_ends:
                    for sentence in self._isentences:
                        sentence.append(self.word2idx(Conll2003Dataset.END_WORD))

                if include_tag_starts:
                    for sentence in self._sentence_itags:
                        sentence.insert(0, self.tag2idx(Conll2003Dataset.START_WORD))

                if include_tag_ends:
                    for sentence in self._sentence_itags:
                        sentence.append(self.tag2idx(Conll2003Dataset.END_WORD))

                self._isentences = [np.array(sentence) for sentence in self._isentences]
                self._sentence_itags = [np.array(sentence) for sentence in self._sentence_itags]
                if os.path.exists(glove_file):
                    self._sentences_embeddings = [np.array(sentence_embeddings)
                                                  for sentence_embeddings
                                                  in self._sentences_embeddings]
                    self._isentences_average = np.array([sentence.mean(axis=0) for sentence in self._sentences_embeddings])
                    self._similarity = (1 / (1 + squareform(
                        pdist(self._isentences_average, metric='cosine')))).mean(axis=0)
                else:
                    self._similarity = None

                if save_file_path is not None:
                    dir_path = Path(save_file_path).parent
                    os.makedirs(dir_path, exist_ok=True)
                    with open(save_file_path, 'wb+') as file:
                        pkl.dump(self, file)

    @property
    def x(self):
        return self._isentences

    @property
    def x_embeddings(self) -> List[np.ndarray]:
        return self._sentences_embeddings

    @property
    def y(self):
        return self._sentence_itags

    @property
    def sim(self):
        return self._similarity

    @property
    def max_word_idx(self) -> int:
        return self._max_word_idx

    @property
    def max_tag_idx(self) -> int:
        return self._max_tag_idx

    def word2idx(self, word: str) -> int:
        return self._word2idx[word]

    def word2vec(self, word: str) -> np.ndarray:
        return self._word2vec[word]

    def tag2idx(self, tag: str) -> int:
        return self._tag2idx[tag]

    def idx2word(self, idx: int) -> str:
        return self._words[idx]

    def idx2vec(self, idx: int) -> np.ndarray:
        return self.word2vec(self.idx2word(idx))

    def idx2tag(self, idx: int) -> str:
        return self._tags[idx]

    def split(self, first_dataset_ratio: float, seed: int = 42) \
            -> Tuple['Conll2003Dataset', 'Conll2003Dataset']:
        np.random.seed(seed)
        isentences_split = train_test_split(self._isentences, test_size=1 - first_dataset_ratio)
        np.random.seed(seed)
        sentence_itags_split = train_test_split(self._sentence_itags,
                                                test_size=1 - first_dataset_ratio)
        if self._similarity is not None:
            np.random.seed(seed)
            sentences_embeddings_split = train_test_split(self._sentences_embeddings, test_size=1 - first_dataset_ratio)
            np.random.seed(seed)
            similarity_split = train_test_split(self._similarity,
                                                test_size=1 - first_dataset_ratio)
        datasets = []
        for i in range(2):
            dataset = Conll2003Dataset(load=False)
            dataset._isentences = isentences_split[i]
            dataset._sentence_itags = sentence_itags_split[i]
            if self._similarity is not None:
                dataset._sentences_embeddings = sentences_embeddings_split[i]
                dataset._similarity = similarity_split[i]
            else:
                dataset._similarity = None
                dataset._sentences_embeddings = None
            dataset._word2idx = self._word2idx
            dataset._tag2idx = self._tag2idx
            dataset._words = self._words
            dataset._tags = self._tags
            datasets.append(dataset)

        return tuple(datasets)

    def train_test_split(self)\
            -> Tuple['Conll2003Dataset', 'Conll2003Dataset']:
        datasets = []
        for i in range(2):
            dataset = Conll2003Dataset(load=False)
            dataset._isentences = self._isentences[i*self.__split_index:(i+1)*self.__split_index]
            dataset._sentence_itags = self._sentence_itags[i*self.__split_index:(i+1)*self.__split_index]
            if self._similarity is not None:
                dataset._similarity = self._similarity[i*self.__split_index:(i+1)*self.__split_index]
                dataset._sentences_embeddings = self._sentences_embeddings[
                                                i * self.__split_index:(i + 1) * self.__split_index]
            else:
                dataset._similarity = None
            dataset._word2idx = self._word2idx
            dataset._tag2idx = self._tag2idx
            dataset._words = self._words
            dataset._tags = self._tags
            dataset._max_word_idx = self._max_word_idx
            dataset._max_tag_idx = self._max_tag_idx
            datasets.append(dataset)

        return tuple(datasets)


if __name__ == '__main__':
    dataset = Conll2003Dataset()
