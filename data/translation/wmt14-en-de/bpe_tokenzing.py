import os

from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer


class BPETokenizer:
    def __init__(self,
                 list_of_sentences,
                 vocab_size=10000):
        self.vocab_size = vocab_size
        self.list_of_sentences = list_of_sentences
        self.pre_tokenizer = AutoTokenizer.from_pretrained("gpt2").backend_tokenizer.pre_tokenizer
        self.word_freqs = self.get_word_freqs()
        self.vocab = self.get_vocab()
        self.splits = self.get_splits()
        self.merges = {}

    def get_word_freqs(self):
        word_freqs = defaultdict(int)
        for sentence in tqdm(self.list_of_sentences):
            words_with_offset = self.pre_tokenizer.pre_tokenize_str(sentence)
            words = [word for word, offset in words_with_offset]
            for word in words:
                word_freqs[word] += 1
        return word_freqs

    def get_vocab(self):
        alphabet = set()
        for word in self.word_freqs.keys():
            for letter in word:
                alphabet.add(letter)
        alphabet = list(alphabet)
        alphabet.sort()
        vocab = ["<endoftext>"] + alphabet.copy()
        return vocab

    def get_splits(self):
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        return splits

    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def train(self):
        pbar = tqdm(total=self.vocab_size - len(self.vocab))
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs()
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges[best_pair] = pair_freqs[best_pair]
            self.merge_pair(*best_pair)
            self.vocab.append(best_pair[0] + best_pair[1])
            pbar.update(1)
        pbar.close()

    def encode(self, sentence):
        words_with_offset = self.pre_tokenizer.pre_tokenize_str(sentence)
        words = [word for word, offset in words_with_offset]
        splits = [[c for c in word] for word in words]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [pair[0] + pair[1]] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])








VOCAB_SIZE = 37000
# read in files
raw_data_path = "data/translation/wmt14-en-de/raw"
train_en_file_path = raw_data_path + "/train/train.en"
train_de_file_path = raw_data_path + "/train/train.de"
with open(train_en_file_path, "r") as f:
    train_en = f.readlines()
with open(train_de_file_path, "r") as f:
    train_de = f.readlines()
corpus = train_en + train_de

bpe_tokenizer = BPETokenizer(corpus, VOCAB_SIZE)
bpe_tokenizer.train()


# write the encode res fro train_en to "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.en"
# write the encode res fro train_de to "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.de"

encoded_train_en_path = "data/translation/wmt14-en-de/tokenized/train/encoded_train.en"
encoded_train_de_path = "data/translation/wmt14-en-de/tokenized/train/encoded_train.de"
# check if the path exists

if not os.path.exists(os.path.dirname(encoded_train_en_path)):
    os.makedirs(os.path.dirname(encoded_train_en_path))

if not os.path.exists(os.path.dirname(encoded_train_de_path)):
    os.makedirs(os.path.dirname(encoded_train_de_path))

with open(encoded_train_en_path, "w") as f:
    for sentence in train_en:
        f.write(" ".join(bpe_tokenizer.encode(sentence)) + "\n")
with open(encoded_train_de_path, "w") as f:
    for sentence in train_de:
        f.write(" ".join(bpe_tokenizer.encode(sentence)) + "\n")


# write the vocab to "../../data/translation/wmt14-en-de/tokenized/vocab"

vocab_path = "../../data/translation/wmt14-en-de/tokenized/vocab"
if not os.path.exists(os.path.dirname(vocab_path)):
    os.makedirs(os.path.dirname(vocab_path))
with open(vocab_path, "w") as f:
    for word in bpe_tokenizer.vocab:
        f.write(word + "\n")
