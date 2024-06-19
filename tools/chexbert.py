import os
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class CheXbert(nn.Module):
    def __init__(self, ckpt_dir, bert_path, checkpoint_path, device, p=0.1):
        super(CheXbert, self).__init__()

        self.ckpt_dir = ckpt_dir
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(bert_path, cache_dir=ckpt_dir)
        config = BertConfig().from_pretrained(bert_path, cache_dir=ckpt_dir)

        with torch.no_grad():

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint
            ckpt_path = os.path.join(self.ckpt_dir, checkpoint_path)
            if not os.path.isfile(ckpt_path):
                raise ValueError(f'The CheXbert checkpoint does not exist at {self.ckpt_dir}, please download it from: https://github.com/stanfordmlgroup/CheXbert#checkpoint-download.')
            state_dict = torch.load(ckpt_path, map_location=device)['model_state_dict']

            new_state_dict = OrderedDict()
            # new_state_dict["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace('module.linear_heads.', 'linear_heads.')
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict)

        self.eval()

    def forward(self, reports):

        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            reports[i] = reports[i].replace("\s+", " ")
            reports[i] = reports[i].replace("\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():

            tokenized = self.tokenizer(
                reports,
                padding='longest',
                return_tensors='pt',
                truncation=True,
                max_length=self.bert.config.max_position_embeddings,
            )

            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)
