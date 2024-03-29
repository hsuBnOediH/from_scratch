from tqdm import tqdm
raw_data_path = "../../data/translation/wmt14-en-de/raw"

train_en_file_path = raw_data_path + "/train/train.en"
train_de_file_path = raw_data_path + "/train/train.de"



with open(train_en_file_path, "r") as f:
    train_en = f.readlines()
with open(train_de_file_path, "r") as f:
    train_de = f.readlines()


from transformers import AutoTokenizer
from collections import defaultdict


tokenizer = AutoTokenizer.from_pretrained("gpt2")

# concat two lists together
croups = train_en + train_de


word_freqs = defaultdict(int)
for sentence in tqdm(croups):
    # [('iron', (0, 4)), ('Ġcement', (4, 11))]
    works_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
    words = [word for word, offset in works_with_offset]
    for word in words:
        word_freqs[word] += 1

alphabet = set()
for word in word_freqs.keys():
    for letter in word:
        alphabet.add(letter)

alphabet = list(alphabet)
alphabet.sort()
print(f"alphabet: {alphabet}")


vocab = ["<endoftext>"] + alphabet.copy()

splits = {word: [c for c in word] for word in word_freqs.keys()}

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs





def merge_pair(a,b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits



merges = {}
VOCAB_SIZE = 500
pbar = tqdm(total=VOCAB_SIZE - len(vocab))
while len(vocab) < VOCAB_SIZE:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or freq > max_freq:
            best_pair = pair
            max_freq = freq
    if max_freq == 1:
        break
    splits = merge_pair(best_pair[0], best_pair[1], splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
    pbar.update(1)
pbar.close()

print(f"merges: {merges}")

def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])