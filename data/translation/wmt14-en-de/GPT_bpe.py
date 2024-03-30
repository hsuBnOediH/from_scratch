from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import pickle


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class BPETokenizer:
    def __init__(self, list_of_sentences, vocab_size=10000):
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
        trie = Trie()
        for word in self.word_freqs.keys():
            for i in range(len(word) - 1):
                pair = word[i:i + 2]
                trie.insert(pair)
        return trie

    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            for i in range(len(word) - 1):
                pair = word[i:i + 2]
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        for word in self.word_freqs:
            i = 0
            while i < len(word) - 1:
                if word[i:i + 2] == (a, b):
                    word = word[:i] + a + b + word[i + 2:]
                else:
                    i += 1
        return word

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
                    if split[i] + split[i + 1] == pair:
                        split = split[:i] + [pair] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])


# Your data paths
VOCAB_SIZE = 500
folder_path = ""
raw_data_path = folder_path + "raw"
train_en_file_path = raw_data_path + "/train/train.en"
train_de_file_path = raw_data_path + "/train/train.de"
encoded_train_en_path = folder_path + "tokenized/train/encoded_train.en"
encoded_train_de_path = folder_path + "tokenized/train/encoded_train.de"
vocab_path = folder_path + "tokenized/vocab"

# Read in files
with open(train_en_file_path, "r") as f:
    train_en = f.readlines()
with open(train_de_file_path, "r") as f:
    train_de = f.readlines()
corpus = train_en + train_de

# Tokenize
bpe_tokenizer = BPETokenizer(corpus[:1000], VOCAB_SIZE)
bpe_tokenizer.train()

# Write encoded data

with open(encoded_train_en_path, "w") as f:
    for sentence in train_en:
        f.write(" ".join(bpe_tokenizer.encode(sentence)) + "\n")
print("Encoded train_en written")
with open(encoded_train_de_path, "w") as f:
    for sentence in train_de[:10]:
        f.write(" ".join(bpe_tokenizer.encode(sentence)) + "\n")
print("Encoded train_de written")

with open(vocab_path, "w") as f:
    for word in bpe_tokenizer.vocab:
        f.write(word + "\n")
print("Vocab written")
