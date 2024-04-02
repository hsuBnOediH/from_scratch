# read in the encoded train_en and train_de from "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.en" and "../../data/translation/wmt14-en-de/tokenized/train/"
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.vanilla_transformer.dataloder import WMT14ENDEDataset, WMT14ENDEDatasetHuggingFace
from models.vanilla_transformer.transformer_structure import TransformerConfig, Transformer
from models.vanilla_transformer.transformer_structure_harvard import EncoderDecoder

tokenizer_folder = "../../data/translation/wmt14-en-de/tokenized"

with open(tokenizer_folder + "/token_to_id.pkl", "rb") as f:
    token_to_id = pickle.load(f)
with open(tokenizer_folder + "/id_to_token.pkl", "rb") as f:
    id_to_token = pickle.load(f)
# convert the encoded train_en and train_de to id

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
device_type = "cuda" if device.type == "cuda" else "mps"

BATCH_SIZE = 32 if device_type == "mps" else 64
SEQ_LEN = 64 if device_type == "mps" else 512
ENCODER_LAYER = 6
DECODER_LAYER = 6
D_MODEL = 256 if device_type == "mps" else 512
HIDDEN_DIM = 512 if device_type == "mps" else 2048
NUM_HEADS = 8
DROPOUT = 0.1
VOCAB_SIZE = len(token_to_id)
EPOCHS = 30
STEPS = 1000000
BETA1 = 0.9
BETA2 = 0.98
EPSILON = 1e-9
LEARNING_RATE = 0.00001
WARMUP_STEPS = 4000


transformer_config = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    encoder_layer=ENCODER_LAYER,
    decoder_layer=DECODER_LAYER,
    d_model=D_MODEL,
    hidden_size=HIDDEN_DIM,
    num_head=NUM_HEADS,
    dropout=DROPOUT,
    vocab_size=VOCAB_SIZE,
    device=device,
    eps = 1e-6,
)
transformer = EncoderDecoder(transformer_config)
transformer.to(device)

# adam with beta1 = 0.9, beta2 = 0.98, epsilon = 1e-9
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPSILON)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()



wmt14_en_de_tokenizer_dataset = WMT14ENDEDatasetHuggingFace(
    en_raw_file_path="../../data/translation/wmt14-en-de/raw/train/train.en",
    de_raw_file_path="../../data/translation/wmt14-en-de/raw/train/train.de",
    device=device, max_len=SEQ_LEN)

dataloader = DataLoader(wmt14_en_de_tokenizer_dataset, batch_size=BATCH_SIZE, shuffle=True)

epoch_loss_list = []
step_loss_list = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # get the batch
        batch_en_tensor = data["en_input_ids"]
        batch_de_tensor = data["de_input_ids"]
        padding_mask_en_tensor = data["en_padding_mask"]
        padding_mask_de_tensor = data["de_padding_mask"]
        # forward pass
        optimizer.zero_grad()
        logit = transformer(batch_en_tensor, batch_de_tensor, padding_mask_en_tensor, padding_mask_de_tensor)
        loss = criterion(logit.view(-1, VOCAB_SIZE), batch_de_tensor.view(-1))
        # backward pass
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
        if step % 30 == 0:
            step_loss_list.append(loss.item())
        if step == STEPS:
            break
    epoch_loss_list.append(epoch_loss)
    print(f"Epoch: {epoch}, Loss: {epoch_loss/len(dataloader)}")


# draw two plots, one for the epoch loss and one for the step loss
import matplotlib.pyplot as plt
plt.plot(epoch_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch Loss")
plt.show()
plt.plot(step_loss_list)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Step Loss")
plt.show()





