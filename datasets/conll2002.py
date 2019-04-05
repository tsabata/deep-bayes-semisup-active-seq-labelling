from typing import Tuple, Optional
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from pathlib import Path


class Conll2002Dataset:
    PADDING = 'PAD'
    UNKNOWN_WORD = 'UNK'
    START_WORD = 'START'
    END_WORD = 'END'

    def __init__(self, load: bool = True,
                 save_file_path: Optional[str] = './datasets/saves/conll2002.pkl',
                 numpy_internal=False, task='NER',
                 include_tag_starts: bool = False, include_tag_ends: bool = False,
                 include_word_starts: bool = False, include_word_ends: bool = False):

        if load:
            if save_file_path is not None and os.path.exists(save_file_path):
                print(f"Loading dataset from {save_file_path}")
                with open(save_file_path, 'rb') as file:
                    dataset = pkl.load(file)
                self.__dict__ = dataset.__dict__
            else:
                nltk.download('conll2002')
                corpus = nltk.corpus.conll2002
                corpus.words()

                tagged_sentences = corpus.iob_sents()
                sentences, pos_tags, ner_tags = zip(*[zip(*ts) for ts in tagged_sentences])

                if task == 'NER':
                    sentence_tags = ner_tags
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

                self._words.insert(0, Conll2002Dataset.UNKNOWN_WORD)
                self._words.insert(0, Conll2002Dataset.END_WORD)
                self._words.insert(0, Conll2002Dataset.START_WORD)
                self._words.insert(0, Conll2002Dataset.PADDING)

                self._tags.insert(0, Conll2002Dataset.UNKNOWN_WORD)
                self._tags.insert(0, Conll2002Dataset.END_WORD)
                self._tags.insert(0, Conll2002Dataset.START_WORD)
                self._tags.insert(0, Conll2002Dataset.PADDING)

                self._max_word_idx = len(self._words)
                self._max_tag_idx = len(self._tags)

                self._word2idx = {word: idx for idx, word in enumerate(self._words)}
                self._tag2idx = {tag: idx for idx, tag in enumerate(self._tags)}

                # Convert input words to integers:
                self._isentences = [[self._word2idx[word] for word in sentence]
                                    for sentence in sentences]

                self._sentence_itags = [[self._tag2idx[word] for word in sentence]
                                        for sentence in sentence_tags]

                if include_word_starts:
                    for sentence in self._isentences:
                        sentence.insert(0, self.word2idx(Conll2002Dataset.START_WORD))

                if include_word_ends:
                    for sentence in self._isentences:
                        sentence.append(self.word2idx(Conll2002Dataset.END_WORD))

                if include_tag_starts:
                    for sentence in self._sentence_itags:
                        sentence.insert(0, self.tag2idx(Conll2002Dataset.START_WORD))

                if include_tag_ends:
                    for sentence in self._sentence_itags:
                        sentence.append(self.tag2idx(Conll2002Dataset.END_WORD))

                if numpy_internal:
                    self._isentences = [np.array(sentence) for sentence in self._isentences]
                    self._sentence_itags = [np.array(sentence) for sentence in self._sentence_itags]

                # self._isentences = [
                #     np.array([self._word2idx[word] for word in sentence])
                #     if numpy_internal
                #     else [self._word2idx[word] for word in sentence]
                #     for sentence in self._isentences]
                # self._sentence_itags = [
                #     np.array([self._tag2idx[tag] for tag in sentence_tags])
                #     if numpy_internal
                #     else
                #     [self._tag2idx[tag] for tag in sentence_tags]
                #     for sentence_tags in sentence_tags]

                if save_file_path is not None:
                    dir_path = Path(save_file_path).parent
                    os.makedirs(dir_path, exist_ok=True)
                    with open(save_file_path, 'wb+') as file:
                        pkl.dump(self, file)

    @property
    def x(self):
        return self._isentences

    @property
    def y(self):
        return self._sentence_itags

    @property
    def max_word_idx(self) -> int:
        return self._max_word_idx

    @property
    def max_tag_idx(self) -> int:
        return self._max_tag_idx

    def word2idx(self, word: str) -> int:
        return self._word2idx[word]

    def tag2idx(self, tag: str) -> int:
        return self._tag2idx[tag]

    def idx2word(self, idx: int) -> str:
        return self._words[idx]

    def split(self, first_dataset_ratio: float, seed: int = 42) \
            -> Tuple['Conll2002Dataset', 'Conll2002Dataset']:
        np.random.seed(seed)
        isentences_split = train_test_split(self._isentences, test_size=1 - first_dataset_ratio)
        np.random.seed(seed)
        sentence_itags_split = train_test_split(self._sentence_itags,
                                                test_size=1 - first_dataset_ratio)
        datasets = []
        for i in range(2):
            dataset = Conll2002Dataset(False)
            dataset._isentences = isentences_split[i]
            dataset._sentence_itags = sentence_itags_split[i]
            dataset._word2idx = self._word2idx
            dataset._tag2idx = self._tag2idx
            dataset._words = self._words
            dataset._tags = self._tags
            datasets.append(dataset)

        return tuple(datasets)
