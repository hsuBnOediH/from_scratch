import os
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.vocab_size)
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = 37000
# read in files
data_folder = "../"
raw_data_path = data_folder + "raw"
train_en_file_path = raw_data_path + "/train/train.en"
train_de_file_path = raw_data_path + "/train/train.de"
with open(train_en_file_path, "r") as f:
    train_en = f.readlines()
with open(train_de_file_path, "r") as f:
    train_de = f.readlines()


tokenized_train_en = []
for idx, sentence in tqdm(enumerate(train_en), total=len(train_en)):
    tokens = tokenizer(sentence, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    tokenized_train_en.append(tokens)
tokenized_train_de = []


# write the encode res for train_en to "../../data/translation/wmt14-en-de/tokenized/train/huggingface_encoded_train.en"
# write the encode res for train_de to "../../data/translation/wmt14-en-de/tokenized/train/huggingface_encoded_train.de"

encoded_train_en_path = data_folder + "tokenized/train/huggingface_encoded_train.en"
# check if the path exists
if not os.path.exists(os.path.dirname(encoded_train_en_path)):
    os.makedirs(os.path.dirname(encoded_train_en_path))
with open(encoded_train_en_path, "wb") as f:
    # write the encoded sentences to the file as pickle
    pickle.dump(tokenized_train_en, f)

for idx, sentence in tqdm(enumerate(train_de), total=len(train_de)):
    tokens = tokenizer(sentence, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    tokenized_train_de.append(tokens)
encoded_train_de_path = data_folder + "tokenized/train/huggingface_encoded_train.de"

if not os.path.exists(os.path.dirname(encoded_train_de_path)):
    os.makedirs(os.path.dirname(encoded_train_de_path))




with open(encoded_train_de_path, "wb") as f:
    # write the encoded sentences to the file as pickle
    pickle.dump(tokenized_train_de, f)


