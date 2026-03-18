import torch.nn as nn
from transformers import RobertaModel, XLNetModel, ElectraModel


class RoBERTaForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.3):
        super(RoBERTaForMultiLabelClassification, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def resize_token_embeddings(self, new_num_tokens):
        self.roberta.resize_token_embeddings(new_num_tokens)


class XLNetForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.3):
        super(XLNetForMultiLabelClassification, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.xlnet.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def resize_token_embeddings(self, new_num_tokens):
        self.xlnet.resize_token_embeddings(new_num_tokens)


class ELECTRAForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.3):
        super(ELECTRAForMultiLabelClassification, self).__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-base-discriminator')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def resize_token_embeddings(self, new_num_tokens):
        self.electra.resize_token_embeddings(new_num_tokens)
