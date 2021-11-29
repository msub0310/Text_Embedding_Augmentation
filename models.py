import torch
from torch import nn, optim

import torchvision.utils

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from attack import *

class BaselineBert(nn.Module):
    
    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME, DROP_OUT_RATE, Config=None):
        super(BaselineBert, self).__init__()
        if Config:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=Config)
        else:        
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        self.drop = nn.Dropout(p=DROP_OUT_RATE)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = encoder_output[0]
        pooled_output = encoder_output[1]

        output = self.drop(pooled_output)
        output = self.out(output)

        return output

class BaselineBertEmbedding(nn.Module):
    
    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME, DROP_OUT_RATE, Config=None):
        super(BaselineBertEmbedding, self).__init__()
        if Config:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=Config)
        else:        
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        self.drop = nn.Dropout(p=DROP_OUT_RATE)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, inputs_embeds, attention_mask):
        encoder_output = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        sequence_output = encoder_output[0]
        pooled_output = encoder_output[1]

        output = self.drop(pooled_output)
        output = self.out(output)

        return output

