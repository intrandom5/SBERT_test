from transformers import AutoModel
import torch.nn as nn
import torch


class SBERT_with_KLUE_BERT(nn.Module):
    def __init__(self):
        super(SBERT_with_KLUE_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained("klue/bert-base")
        # klue/bert-base 모델의 output shape이 768 차원이므로,
        self.linear = nn.Linear(768*2, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids).last_hidden_state
        v = self.bert(tgt_ids).last_hidden_state
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)
        attn_outputs = torch.cat((u, v), dim=-1)
        outputs = self.linear(attn_outputs)

        # u = self.bert(src_ids).last_hidden_state
        # v = self.bert(tgt_ids).last_hidden_state
        # u = torch.mean(u, dim=1)
        # v = torch.mean(v, dim=1)
        # outputs = self.similarity(u, v)

        return outputs

class SBERT_with_ROBERTA_LARGE(nn.Module):
    def __init__(self):
        super(SBERT_with_ROBERTA_LARGE, self).__init__()
        self.bert = AutoModel.from_pretrained("klue/roberta-large")
        self.linear = nn.Linear(1024*2, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids).last_hidden_state
        v = self.bert(tgt_ids).last_hidden_state
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)
        attn_outputs = torch.cat((u, v), dim=-1)
        outputs = self.linear(attn_outputs)
        # u = self.bert(src_ids).last_hidden_state
        # v = self.bert(tgt_ids).last_hidden_state
        # u = torch.mean(u, dim=1)
        # v = torch.mean(v, dim=1)
        # outputs = self.similarity(u, v)

        return outputs

class SBERT_with_KOELECTRA_BASE(nn.Module):
    def __init__(self, version):
        super(SBERT_with_KOELECTRA_BASE, self).__init__()
        if version == "v2":
            self.bert = AutoModel.from_pretrained("monologg/koelectra-base-v2-discriminator")
        elif version == "v3":
            self.bert = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        else:
            self.bert = AutoModel.from_pretrained("monologg/koelectra-base-discriminator")
        self.linear = nn.Linear(768*2, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids).last_hidden_state
        v = self.bert(tgt_ids).last_hidden_state
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)
        attn_outputs = torch.cat((u, v), dim=-1)
        outputs = self.linear(attn_outputs)
        # u = self.bert(src_ids).last_hidden_state
        # v = self.bert(tgt_ids).last_hidden_state
        # u = torch.mean(u, dim=1)
        # v = torch.mean(v, dim=1)
        # outputs = self.similarity(u, v)

        return outputs
