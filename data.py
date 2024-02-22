import os
from io import open
import torch

from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def len(self):
        return self.seq_len

    def __init__(self, path):
        self.seq_len = 0
        self.dictionary = Dictionary()
        # 添加一个参数，记录设置的最大序列长度
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # 改成相应的文件名
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def pad(self, lines):
        pad = self.dictionary.word2idx['<pad>']
        ret = []
        for (ids, label) in lines:
            for i in range(len(ids), self.seq_len):
                ids.append(pad)
            ret.append((ids, label))
        return ret

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        self.dictionary.add_word('<pad>')  # 用于padding
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                [line, tag] = line.split('\t')
                words = line.split()  # 不再需要<eos>了。为什么？<eos>在LM任务中起到什么作用？
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            labels = []
            ###############################################################################
            # sst2按行构建输入, 长于seq_len的句子进行截断，短于seq_len的用<pad>补齐长度至seq_len
            ###############################################################################
            seq_len = 0
            lines = []
            for line in f:
                [line, tag] = line.split('\t')
                if tag == "positive\n":
                    sentiment_label = 1
                elif tag == "negative\n":
                    sentiment_label = 0
                else:
                    raise "bad data"
                labels.append(torch.tensor([sentiment_label]).float().unsqueeze(0))
                words = line.split()
                ids = [self.dictionary.word2idx[word] for word in words]
                if len(ids) > seq_len:
                    seq_len = len(ids)
                line_data = (ids, sentiment_label)
                lines.append(line_data)
            if self.seq_len < seq_len:
                self.seq_len = seq_len
        return lines
