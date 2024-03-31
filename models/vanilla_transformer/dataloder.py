import torch
import torch.utils.data as data


class WMT14ENDEDataset(data.Dataset):

    def __init__(self, en_token_file_path= "",de_token_file_path="",
               max_len=512, token_to_id=None, device="cuda"):
        self.device = device
        with open(en_token_file_path, "r") as f:
            en_token_ids = f.readlines()
        with open(de_token_file_path, "r") as f:
            de_token_ids = f.readlines()
        assert len(en_token_ids) == len(de_token_ids), "The number of english and german sentences should be the same"
        self.data = list(zip(en_token_ids, de_token_ids))
        self.max_len = max_len
        self.token_to_id = token_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en_sentence,de_sentence = self.data[idx]
        if len(en_sentence.strip()) == 0:
            en_sentence = "<pad>"
        if len(de_sentence.strip()) == 0:
            de_sentence = "<pad>"
        en_sentence_id = [self.token_to_id[token] for token in en_sentence.strip().split()]
        de_sentence_id = [self.token_to_id[token] for token in de_sentence.strip().split()]
        en_padding_mask = [1] * len(en_sentence_id)
        de_padding_mask = [1] * len(de_sentence_id)
        if len(en_sentence_id) > self.max_len:
            en_sentence_id = en_sentence_id[:self.max_len]
            en_padding_mask = en_padding_mask[:self.max_len]
        if len(de_sentence_id) > self.max_len:
            de_sentence_id = de_sentence_id[:self.max_len]
            de_padding_mask = de_padding_mask[:self.max_len]
        if len(en_sentence_id) < self.max_len:
            en_sentence_id += [self.token_to_id["<pad>"]] * (self.max_len - len(en_sentence_id))
            en_padding_mask += [0] * (self.max_len - len(en_padding_mask))
        if len(de_sentence_id) < self.max_len:
            de_sentence_id += [self.token_to_id["<pad>"]] * (self.max_len - len(de_sentence_id))
            de_padding_mask += [0] * (self.max_len - len(de_padding_mask))


        return {
            "en_input_ids": torch.tensor(en_sentence_id, device=self.device),
            "de_input_ids": torch.tensor(de_sentence_id, device=self.device),
            "en_padding_mask": torch.tensor(en_padding_mask, device=self.device),
            "de_padding_mask": torch.tensor(de_padding_mask, device=self.device)
        }