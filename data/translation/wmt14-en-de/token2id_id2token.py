# read the vocab from "../../data/translation/wmt14-en-de/tokenized/vocab"
vocab_path = "../../data/translation/wmt14-en-de/tokenized/vocab"
token_to_id = {}
id_to_token = {}
# add padding token
token_to_id["<pad>"] = 0
id_to_token[0] = "<pad>"

with open(vocab_path, "r") as f:
    for idx, line in enumerate(f):
        token = line.strip()
        token_to_id[token] = idx
        id_to_token[idx] = token

# write token_to_id and id_to_token as pickle files
import pickle
token_to_id_path = "../../data/translation/wmt14-en-de/tokenized/token_to_id.pkl"
id_to_token_path = "../../data/translation/wmt14-en-de/tokenized/id_to_token.pkl"
with open(token_to_id_path, "wb") as f:
    pickle.dump(token_to_id, f)
with open(id_to_token_path, "wb") as f:
    pickle.dump(id_to_token, f)