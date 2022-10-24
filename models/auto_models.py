from transformers import AutoModel
import torch.nn as nn
import torch


class SBERT_with_KLUE_BERT(nn.Module):
    def __init__(self):
        super(SBERT_with_KLUE_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained("klue/bert-base")
        # klue/bert-base 모델의 output shape이 768 차원이므로,
        self.linear = nn.Linear(768*2, 1)

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids)['pooler_output']
        v = self.bert(tgt_ids)['pooler_output']
        
        attn_outputs = torch.cat((u, v), dim=-1)
        outputs = self.linear(attn_outputs)

        return outputs
