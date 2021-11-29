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
from datasets  import *
from attack import *

model_name = "ADV_EMBEDDING_BATCH64_EPS0.01_ALPHA0.000125_STEP2_AUG5_START12_gener_best"
print(model_name)

RANDOM_SEED = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_seed(RANDOM_SEED)

#generate same adversarial example
torch.backends.cudnn.deterministic = True
#

#params
MAX_LEN = 512
BATCH_SIZE = 8
DROP_OUT_RATE = 0.3
EPOCHS = 30
WARM_UP_STEPS = 0
LEARNING_RATE = 2e-5

#after (BATCH_SIZE * ACCUMULATION_STEPS) steps -> update
ACCUMULATION_STEPS = 8

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

data_path = "./data/IMDB_Dataset"
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

if not str(MAX_LEN) in str(os.listdir(f"{data_path}/")):
    encoding_train, encoding_val, encoding_test = tokenizing_dataset(df_train, df_val, df_test, tokenizer, MAX_LEN, data_path)
else:
    encoding_train, encoding_val, encoding_test = load_tokenized_dataset(data_path, MAX_LEN)

train_data_loader = create_data_loader(df_train, encoding_train, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
val_data_loader = create_data_loader(df_val, encoding_val, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)
test_data_loader = create_data_loader(df_test, encoding_test, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)

model = BaselineBertEmbedding(n_classes, PRE_TRAINED_MODEL_NAME, DROP_OUT_RATE)    

model.load_state_dict(torch.load(f"./results/{model_name}/best_model_state.bin", map_location=device))
#model.load_state_dict(torch.load(f"./results/{model_name}/last_model_state_epoch30.bin", map_location=device))

model = model.to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

test_acc, _ = eval_model_embedding(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)

print(test_acc.item())