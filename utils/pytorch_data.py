from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch


class Dictionary(object):
    def __init__(self, oov = '<unk>'):
        self.word2idx = {}
        self.idx2word = []
        self.oov = oov

    def read_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.split()
                # assert (len(word) == 2)
                word = word[0]
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path = '/vol/tensusers/mbentum/FRISIAN_ASR/LM/', oov = '<unk>'):
        self.dictionary = Dictionary(oov)
        self.dictionary.read_vocab(os.path.join(path, 'vocab_council'))
        self.train = self.tokenize(os.path.join(path, 'council_notes_manual_transcriptions_train'))
        self.valid = self.tokenize(os.path.join(path, 'manual_transcriptions_dev'))
        self.test = self.tokenize(os.path.join(path, 'manual_transcriptions_test'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as f:
            all_ids = []
            for line in f:
                words = line.split() + ['<s>']
                ids = []
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids.append(self.dictionary.word2idx[word])
                    else:
                        ids.append(self.dictionary.word2idx[self.dictionary.oov])
                all_ids.append(torch.tensor(ids).type(torch.int64))
            data = torch.cat(all_ids)

        return data


