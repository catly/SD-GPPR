# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from transformers import BertModel


#
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        # self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.mid = nn.Linear(config.hidden_size, config.mid_size)
        self.fc = nn.Linear(config.mid_size, config.class_num)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        mid_out = self.mid(pooled_output)
        out = self.fc(mid_out)
        return out, mid_out
