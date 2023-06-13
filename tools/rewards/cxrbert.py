import importlib
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer


class CXRBERTReward:

    def __init__(self, device):
        self.device = device

        # Load the model and tokenizer:
        ckpt_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def __call__(self, predictions, labels):
        return self.reward(predictions, labels)

    def reward(self, predictions, labels):
        assert isinstance(predictions, list), '"predictions" must be a list of strings.'
        assert all(isinstance(i, str) for i in predictions), 'Each element of "predictions" must be a string.'
        assert isinstance(labels, list), '"labels" must be a list of lists, where each sub-list has a multiple strings.'
        assert all(isinstance(i, list) for i in labels), 'Each element of "labels" must be a list of strings.'
        assert all(isinstance(j, str) for i in labels for j in i), 'each sub-list must have one or more strings.'

        with torch.no_grad():

            # Tokenize and compute the sentence embeddings:
            tokenizer_output = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=predictions,
                add_special_tokens=True,
                padding='longest',
                return_tensors='pt',
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            )

            prediction_embeddings = self.model(
                input_ids=tokenizer_output.input_ids.to(self.device),
                attention_mask=tokenizer_output.attention_mask.to(self.device),
                output_cls_projected_embedding=True,
                return_dict=True,
            )

            tokenizer_output = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[j for i in labels for j in i],
                add_special_tokens=True,
                padding='longest',
                return_tensors='pt',
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            )

            label_embeddings = self.model(
                input_ids=tokenizer_output.input_ids.to(self.device),
                attention_mask=tokenizer_output.attention_mask.to(self.device),
                output_cls_projected_embedding=True,
                return_dict=True,
            )

            # Compute the cosine similarity of sentence embeddings obtained from input text prompts.
            sim = torch.nn.functional.cosine_similarity(
                prediction_embeddings.cls_projected_embedding,
                label_embeddings.cls_projected_embedding,
            )

        return sim
