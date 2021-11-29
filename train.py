import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import os
from tqdm import tqdm
import pickle
import random
from utils import *
from models import *
from datasets import *

RANDOM_SEED = 42
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
set_seed(RANDOM_SEED)

#params
MAX_LEN = 512
BATCH_SIZE = 8
DROP_OUT_RATE = 0.3
EPOCHS = 30
WARM_UP_STEPS = 0

#after (BATCH_SIZE * ACCUMULATION_STEPS) steps -> update
ACCUMULATION_STEPS = 2

CUSTOM_MODEL_NAME = "PGD_BATCH16"

params = f"max len : {MAX_LEN}\nbatch size : {BATCH_SIZE}\ndropout rate : {DROP_OUT_RATE}\nepochs : {EPOCHS}\nwarm up steps : {WARM_UP_STEPS}\naccumulation steps : {ACCUMULATION_STEPS}\nmodel name : {CUSTOM_MODEL_NAME}\n"

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

save_path = f"./results/{CUSTOM_MODEL_NAME}"
data_path = "./data/IMDB_Dataset"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#main
df = pd.read_csv(f"{data_path}/IMDB_Dataset.csv")
df = preprocess_rating(df)

n_classes = len(df["sentiment"].unique())

df_train, df_test = train_test_split(
    df,
    test_size=0.6,
    random_state=RANDOM_SEED
)

df_val, df_test = train_test_split(
    df_test,
    test_size=(5/6),
    random_state=RANDOM_SEED
)

params += f"train : {len(df_train)}\nvalidation : {len(df_val)}\ntest : {len(df_test)}\n"
print(params)

if not str(MAX_LEN) in str(os.listdir(f"{data_path}/")):
    encoding_train, encoding_val, encoding_test = tokenizing_dataset(df_train, df_val, df_test, tokenizer, MAX_LEN, data_path)
else:
    encoding_train, encoding_val, encoding_test = load_tokenized_dataset(data_path, MAX_LEN)

train_data_loader = create_data_loader(df_train, encoding_train, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
val_data_loader = create_data_loader(df_val, encoding_val, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)
test_data_loader = create_data_loader(df_test, encoding_test, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)

model = BaselineBert(n_classes, PRE_TRAINED_MODEL_NAME, DROP_OUT_RATE)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARM_UP_STEPS,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train),
        ACCUMULATION_STEPS
    )

    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history["train_acc"].append(train_acc.cpu())
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc.cpu())
    history["val_loss"].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), save_path + "/best_model_state.bin")
        best_accuracy = val_acc

print("training finished")

print(" ")

save_history_params(history, params, save_path)