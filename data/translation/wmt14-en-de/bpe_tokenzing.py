import os
import pickle

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
        self.init_vocab_size = len(self.vocab)
        self.splits = self.get_splits()
        self.merges = {}

    def get_word_freqs(self):
        word_freqs_folder_path = "tokenized/word_freqs"
        if not os.path.exists(word_freqs_folder_path):
            os.makedirs(word_freqs_folder_path)
        word_freqs_path = word_freqs_folder_path + "/word_freqs.pkl"
        if os.path.exists(word_freqs_path):
            with open(word_freqs_path, "rb") as f:
                word_freqs = pickle.load(f)
                return word_freqs

        word_freqs = defaultdict(int)
        for sentence in tqdm(self.list_of_sentences):
            words_with_offset = self.pre_tokenizer.pre_tokenize_str(sentence)
            words = [word for word, offset in words_with_offset]
            for word in words:
                word_freqs[word] += 1
        with open(word_freqs_path, "wb") as f:
            pickle.dump(word_freqs, f)
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
        # if have checkpoint pickles, load it, otherwise start from scratch
        checkpoint_folder_path = "tokenized/checkpoint"
        # if the checkpoint folder does not exist, create it
        if not os.path.exists(checkpoint_folder_path):
            os.makedirs(checkpoint_folder_path)
        # if the folder is empty, start from scratch, otherwise load the file have the highest number
        checkpoint_files = os.listdir(checkpoint_folder_path)
        # remove the files that are not checkpoint files
        for file in checkpoint_files:
            if not file.split(".")[0].isdigit():
                checkpoint_files.remove(file)
        if checkpoint_files:
            latest_checkpoint_file = None
            for file in checkpoint_files:
                # the file name is not a number, skip it
                if not file.split(".")[0].isdigit():
                    continue
                if not latest_checkpoint_file:
                    latest_checkpoint_file = file
                else:
                    if int(file.split(".")[0]) > int(latest_checkpoint_file.split(".")[0]):
                        latest_checkpoint_file = file
            # load the latest checkpoint file
            with open(checkpoint_folder_path + "/" + latest_checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
                self.vocab = checkpoint["vocab"]
                self.splits = checkpoint["splits"]
                self.merges = checkpoint["merges"]

        pbar = tqdm(total=self.vocab_size - self.init_vocab_size)
        # update the progress bar according to the number of vocab
        pbar.update(len(self.vocab) - self.init_vocab_size)

        while len(self.vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs()
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges[best_pair] = pair_freqs[best_pair]
            self.merge_pair(*best_pair)
            self.vocab.append(best_pair[0] + best_pair[1])
            pbar.update(1)
            if len(self.vocab) % 500 == 0:
                checkpoint = {
                    "vocab": self.vocab,
                    "splits": self.splits,
                    "merges": self.merges
                }
                with open(checkpoint_folder_path + "/" + str(len(self.vocab)) + ".pkl", "wb") as f:
                    pickle.dump(checkpoint, f)
            # if the number of files in the checkpoint folder is greater than 10, delete the file with the smallest number
            checkpoint_files = os.listdir(checkpoint_folder_path)
            if len(checkpoint_files) > 10:
                smallest_checkpoint_file = None
                for file in checkpoint_files:
                    if not file.split(".")[0].isdigit():
                        continue
                    if not smallest_checkpoint_file:
                        smallest_checkpoint_file = file
                    else:
                        if int(file.split(".")[0]) < int(smallest_checkpoint_file.split(".")[0]):
                            smallest_checkpoint_file = file
                os.remove(checkpoint_folder_path + "/" + smallest_checkpoint_file)
                checkpoint_files.remove(smallest_checkpoint_file)
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
data_folder = ""
raw_data_path = data_folder + "raw"
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

encoded_train_en_path = "tokenized/train/encoded_train.en"
encoded_train_de_path = "tokenized/train/encoded_train.de"
# check if the path exists

if not os.path.exists(os.path.dirname(encoded_train_en_path)):
    os.makedirs(os.path.dirname(encoded_train_en_path))

if not os.path.exists(os.path.dirname(encoded_train_de_path)):
    os.makedirs(os.path.dirname(encoded_train_de_path))

print("writing encoded train_en and train_de to file")
tokenized_train_en = []

for sentence in tqdm(train_en):
    tokenized_train_en.append(" ".join(bpe_tokenizer.encode(sentence)) + "\n")

with open(encoded_train_en_path, "w") as f:
    for sentence in tqdm(tokenized_train_en):
        f.write(sentence)
print("finished writing encoded train_en to file")
print("writing encoded train_de to file")
tokenized_train_de = []
# using multiprocessing to speed up the process


for sentence in tqdm(train_de):
    tokenized_train_de.append(" ".join(bpe_tokenizer.encode(sentence)) + "\n")

with open(encoded_train_de_path, "w") as f:
    for sentence in tqdm(tokenized_train_de):
        f.write(sentence)
print("finished writing encoded train_de to file")

vocab_path = "tokenized/vocab"
if not os.path.exists(os.path.dirname(vocab_path)):
    os.makedirs(os.path.dirname(vocab_path))
print("writing vocab to file")
with open(vocab_path, "w") as f:
    for word in tqdm(bpe_tokenizer.vocab):
        f.write(word + "\n")
print("finished writing vocab to file")
