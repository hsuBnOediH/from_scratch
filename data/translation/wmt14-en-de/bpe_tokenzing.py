import os
VOCAB_SIZE = 37000
# read in files
raw_data_path = "../../data/translation/wmt14-en-de/raw"
train_en_file_path = raw_data_path + "/train/train.en"
train_de_file_path = raw_data_path + "/train/train.de"
with open(train_en_file_path, "r") as f:
    train_en = f.readlines()
with open(train_de_file_path, "r") as f:
    train_de = f.readlines()
corpus = train_en + train_de
from tokenizer.bpe.BPE import BPETokenizer

bpe_tokenizer = BPETokenizer(corpus, VOCAB_SIZE)
bpe_tokenizer.train()


# write the encode res fro train_en to "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.en"
# write the encode res fro train_de to "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.de"

encoded_train_en_path = "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.en"
encoded_train_de_path = "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.de"
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
