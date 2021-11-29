import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
import pickle

import gc

def sentiment_to_rating(sentiment):
    if sentiment == "positive":
        return 1

    elif sentiment == "negative":
        return 0

    else:
        return sentiment

def preprocess_rating(df):
    
    df["sentiment"] = df["sentiment"].apply(lambda x: sentiment_to_rating(x))

    return df

class TextDataset(Dataset):
    
    def __init__(self, reviews, targets, encodings, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.encodings[item]

        return {
            "review_text" : review,
            "input_ids" : encoding["input_ids"].flatten(),
            "attention_mask" : encoding["attention_mask"].flatten(),
            "token_type_ids" : encoding["token_type_ids"].flatten(),
            "targets" : torch.tensor(target, dtype=torch.long)
            }

def create_data_loader(df, encodings, tokenizer, max_len, batch_size, shuffle):
    ds = TextDataset(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        encodings=encodings,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle
    )

def tokenizing(reviews, tokenizer, max_len):
    encoded_text = []
    
    for review in tqdm(reviews):
    
        encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=True,
            padding="max_length",
            truncation="longest_first",
            return_attention_mask=True,
            return_tensors="pt"
        )
    
        encoded_text.append(encoding)
    
    return encoded_text

def tokenizing_dataset(df_train, df_val, df_test, tokenizer, max_len, data_path):
    print("start tokenizing")
    print(" ")
    
    tokenized_train = tokenizing(df_train.review, tokenizer, max_len)
    with open(f"{data_path}/tokenized_train_{max_len}.pickle", "wb") as f:
        pickle.dump(tokenized_train, f, pickle.HIGHEST_PROTOCOL)
    print("tokenized_train saved")

    tokenized_val = tokenizing(df_val.review, tokenizer, max_len)
    with open(f"{data_path}/tokenized_val_{max_len}.pickle", "wb") as f:
        pickle.dump(tokenized_val, f, pickle.HIGHEST_PROTOCOL)
    print("tokenized_val saved")

    tokenized_test = tokenizing(df_test.review, tokenizer, max_len)
    with open(f"{data_path}/tokenized_test_{max_len}.pickle", "wb") as f:
        pickle.dump(tokenized_test, f, pickle.HIGHEST_PROTOCOL)
    print("tokenized_test saved")
    print(" ")

    print("finised tokenizing")
    print(" ")

    return tokenized_train, tokenized_val, tokenized_test

def tokenizing_two_sentence(sentences_1, sentences_2, tokenizer, max_len):
    encoded_text = []

    if len(sentences_1) != len(sentences_2):
        raise("Length of two sentences are different")

    for index in tqdm(range(len(sentences_1))):
        sentence_1 = sentences_1[index]
        sentence_2 = sentences_2[index]

        encoding = tokenizer.encode_plus(
            sentence_1,
            sentence_2,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=True,
            padding="max_length",
            truncation="longest_first",
            return_attention_mask=True,
            return_tensors="pt"
        )

        encoded_text.append(encoding)

    return encoded_text

def tokenizing_two_sentence_dataset(df_train, df_val, df_test, tokenizer, max_len, data_path, columns):
    print("start tokenizing")
    print(" ")

    tokenized_train = tokenizing_two_sentence(df_train[columns[0]], df_train[columns[1]], tokenizer, max_len)
    with open(f"{data_path}/tokenized_train_{max_len}.pickle", "wb") as f:
        pickle.dump(tokenized_train, f, pickle.HIGHEST_PROTOCOL)
    print("tokenized_train saved")

    tokenized_val = tokenizing_two_sentence(df_val[columns[0]], df_val[columns[1]], tokenizer, max_len)
    with open(f"{data_path}/tokenized_val_{max_len}.pickle", "wb") as f:
        pickle.dump(tokenized_val, f, pickle.HIGHEST_PROTOCOL)
    print("tokenized_val saved")

    tokenized_test = tokenizing_two_sentence(df_test[columns[0]], df_test[columns[1]], tokenizer, max_len)
    with open(f"{data_path}/tokenized_test_{max_len}.pickle", "wb") as f:
        pickle.dump(tokenized_test, f, pickle.HIGHEST_PROTOCOL)
    print("tokenized_test saved")
    print(" ")

    print("finised tokenizing")
    print(" ")

    return tokenized_train, tokenized_val, tokenized_test

def load_tokenized_dataset(data_path, max_len):
    with open(f"{data_path}/tokenized_train_{max_len}.pickle", "rb") as f:
        tokenized_train = pickle.load(f)

    with open(f"{data_path}/tokenized_val_{max_len}.pickle", "rb") as f:
        tokenized_val = pickle.load(f)

    with open(f"{data_path}/tokenized_test_{max_len}.pickle", "rb") as f:
        tokenized_test = pickle.load(f)

    print("finished loading tokenized data")
    print(" ")

    return tokenized_train, tokenized_val, tokenized_test

def generate_adversarial_masks(model, adversarial_embeddings, device):
    
    pad_token = torch.tensor(0).to(device)

    pad_embedding = model.bert.embeddings.word_embeddings(pad_token).to("cpu")

    adversarial_masks = 1 - (adversarial_embeddings == pad_embedding).all(dim=2).float()

    del(pad_embedding)

    return adversarial_masks

class AdversarialDataset(Dataset):
    
    def __init__(self, adversarial_embeddings, adversarial_labels, adversarial_masks, max_len):
        self.embeddings = adversarial_embeddings
        self.targets = adversarial_labels
        self.attention_masks = adversarial_masks
        self.max_len = max_len

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        embedding = self.embeddings[item]
        target = self.targets[item]
        attention_mask = self.attention_masks[item]

        return {
            "embeddings" : embedding,
            "attention_masks" : attention_mask,
            "targets" : target
        }

def create_adv_loader(adversarial_embeddings, adversarial_labels, adversarial_masks, max_len, batch_size, shuffle):
    ds = AdversarialDataset(
        adversarial_embeddings=adversarial_embeddings,
        adversarial_labels=adversarial_labels,
        adversarial_masks=adversarial_masks,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

def preprocessing_two_sentences(df, columns, label_name, targets):
    
    df[targets[0]] = df.apply(lambda x: x[columns[0]] + "/APPEND/" + x[columns[1]], axis=1)
    df[targets[1]] = df[columns[2]].apply(lambda x: 0 if x == label_name else 1)

    df = df[targets]

    return df