# read in the encoded train_en and train_de from "../../data/translation/wmt14-en-de/tokenized/train/encoded_train.en" and "../../data/translation/wmt14-en-de/tokenized/train/"
import os.path
import random
from datetime import time

import numpy as np
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from transformer_structure import *
from dataloder import *
from eval.blue import compute_bleu


# REPORT_WANDB = True
REPORT_WANDB = False
run_name = "self_implemented_transformer_not_converging"

tokenizer_folder = "../../data/translation/wmt14-en-de/tokenized"
check_point_folfer_path = "checkpoint/"

# with open(tokenizer_folder + "/token_to_id.pkl", "rb") as f:
#     token_to_id = pickle.load(f)
# with open(tokenizer_folder + "/id_to_token.pkl", "rb") as f:
#     id_to_token = pickle.load(f)
# convert the encoded train_en and train_de to id

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
device_type = "cuda" if device.type == "cuda" else "mps"

BATCH_SIZE = 32 if device_type == "mps" else 64
SEQ_LEN = 64 if device_type == "mps" else 512
ENCODER_LAYER_NUM = 6
DECODER_LAYER_NUM = 6
D_MODEL = 256 if device_type == "mps" else 512
HIDDEN_DIM = 512 if device_type == "mps" else 2048
NUM_HEADS = 4
DROPOUT = 0.1
# VOCAB_SIZE = len(token_to_id)
VOCAB_SIZE = 50256
EPOCHS = 50
STEPS = 1000000
BETA1 = 0.9
BETA2 = 0.98
EPSILON = 1e-9
LEARNING_RATE = 0.00001
WARMUP_STEPS = 4000

seed_value = 42
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

if not os.path.exists(check_point_folfer_path):
    os.makedirs(check_point_folfer_path)

checkpoint_files = os.listdir(check_point_folfer_path)
checkpoint_file_name = f"checkpoint_1000_batch_size-{BATCH_SIZE}_seq_len-{SEQ_LEN}_encoder_layer_num-{ENCODER_LAYER_NUM}_decoder_layer_num-{DECODER_LAYER_NUM}_d_model-{D_MODEL}_hidden_dim-{HIDDEN_DIM}_num_heads-{NUM_HEADS}_dropout-{DROPOUT}_vocab_size-{VOCAB_SIZE}_epochs-{EPOCHS}_steps-{STEPS}_beta1-{BETA1}_beta2-{BETA2}_epsilon-{EPSILON}_learning_rate-{LEARNING_RATE}_warmup_steps-{WARMUP_STEPS}"
if checkpoint_file_name in checkpoint_files:
    # load the model from the checkpoint
    transformer = torch.load(check_point_folfer_path + "/" + checkpoint_file_name)
else:


    if REPORT_WANDB:
        wandb.init(
        # set the wandb project where this run will be logged
        project="from_scratch_vanilla_transformer_debug",
        name=run_name,

        # track hyperparameters and run metadata
        config={
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "encoder_layer_num": ENCODER_LAYER_NUM,
            "decoder_layer_num": DECODER_LAYER_NUM,
            "d_model": D_MODEL,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "vocab_size": VOCAB_SIZE,
            "epochs": EPOCHS,
            "steps": STEPS,
            "beta1": BETA1,
            "beta2": BETA2,
            "epsilon": EPSILON,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "device": device.type,
            "timestamp": time()
        }
        )

    transformer_config = TransformerConfig(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        encoder_layer_num=ENCODER_LAYER_NUM,
        decoder_layer_num=DECODER_LAYER_NUM,
        d_model=D_MODEL,
        d_ff=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        vocab_size=VOCAB_SIZE,
        device=device,
        eps = 1e-6,
    )
    transformer = Transformer(transformer_config)
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
        if REPORT_WANDB:
            wandb.log({"epoch": epoch, "loss": epoch_loss/len(dataloader)})

    # save the model
    torch.save(transformer, check_point_folfer_path + "/" + checkpoint_file_name)

def transfomer_inference(batch_src_tensor, padding_mask_src_tensor, max_len=SEQ_LEN):
    # feed the src tensor, padding mask tensor and tgt tensor including the bos token to the transformer
    # loop until the model generate the eos token or the length of the tgt tensor is equal to max_len
    # return the tgt tensor
    tgt_tensor = torch.tensor([[tokenizer.bos_token_id]] * len(batch_src_tensor), device=device)
    tgt_mask = torch.ones_like(tgt_tensor)
    for _ in range(max_len):
        logit = transformer(batch_src_tensor, tgt_tensor, padding_mask_src_tensor, tgt_mask)
        logit = torch.softmax(logit, dim=-1)
        pred_sents_ids = torch.argmax(logit, dim=-1)
        tgt_tensor = torch.cat([tgt_tensor, pred_sents_ids], dim=-1)
        tgt_mask = torch.ones_like(tgt_tensor)
        if pred_sents_ids[-1] == tokenizer.eos_token_id:
            break
    return tgt_tensor



# evaluate the model using bleu score
transformer.eval()
test_dataset = WMT14ENDEDatasetHuggingFace(
    en_raw_file_path="../../data/translation/wmt14-en-de/raw/test/newstest2012.en",
    de_raw_file_path="../../data/translation/wmt14-en-de/raw/test/newstest2012.de",
    device=device, max_len=SEQ_LEN)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

ref_sents_list = []
pred_sents_list = []
tokenizer = test_dataset.tokenizer
for step, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    batch_en_tensor = data["en_input_ids"]
    padding_mask_en_tensor = data["en_padding_mask"]
    ref_sents = data["de_sentence_str"]
    # prepare the decoder input and padding mask for inference
    res = transfomer_inference(batch_en_tensor, padding_mask_en_tensor)
    # softmax the logit and move it to cpu
    logit = torch.softmax(logit, dim=-1).cpu()
    pred_sents_ids = torch.argmax(logit, dim=-1)
    pred_sents = pred_sents_ids.tolist()
    decoded_sents = tokenizer.batch_decode(pred_sents,remove_special_tokens=True)
    # extend the the pred_sents_list
    pred_sents_list.extend(decoded_sents)
    # for each ele of ref_sents, put in list and append to ref_sents_list
    ref_sents_list.extend([[ref_sent]for ref_sent in ref_sents])
blue_score = compute_bleu(ref_sents_list, pred_sents_list, smooth=True, max_order=4)
print(f"BLEU Score: {blue_score}")

if REPORT_WANDB:
    wandb.finish()



