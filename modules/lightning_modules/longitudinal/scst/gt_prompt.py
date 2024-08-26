import torch
import transformers

from modules.lightning_modules.longitudinal.gt_prompt import GTPrompt
from tools.rewards.cxrbert import CXRBERTReward


class SCSTGTPrompt(GTPrompt):

    def __init__(
        self,
        trial,
        scst_sample_top_p: float = 1.0,
        scst_sample_top_k: float = 50,
        scst_sample_temperature: float = 1.0,
        **kwargs,
    ):
        """
        Argument/s:
            trial - trial number for the model.
            scst_sample_top_p - only the most probable tokens with probabilities that add up to top_p or higher are
                considered during sampling.
            scst_sample_top_k - only the top-k ranked tokens are considered during sampling.
            scst_sample_temperature - the sharpness of the softmax probability distribution during sampling.
            kwargs - keyword arguments.
        """
        super(SCSTGTPrompt, self).__init__(**kwargs)

        self.trial = trial
        self.scst_sample_top_p = scst_sample_top_p
        self.scst_sample_top_k = scst_sample_top_k
        self.scst_sample_temperature = scst_sample_temperature
 
        # Freeze the encoder:
        for p in self.encoder_decoder.encoder.parameters():
            p.requires_grad = False

        # Unfreeze all parameters of the decoder (even LoRA):
        for p in self.encoder_decoder.decoder.parameters():
            p.requires_grad = True

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # SCST step:
        loss = self.scst_step(batch, batch_idx)

        # Logging:
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric:
        return loss

    def scst_step(self, batch, batch_idx):
        """
        Self-critical sequence training step (the training loss needs to be returned).

        Argument/s:
            batch - mini-batch from the training set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.

        Returns:
            loss - training loss for the mini-batch.
        """

        # Tokenize prompt:
        prompt = self.encoder_decoder.tokenize_prompt(
            batch['previous_findings'], 
            batch['previous_impression'],             
            self.tokenizer, 
            self.decoder_max_len,  
            add_bos_token_id=True,
        )

        # Encoder outputs:
        encoder_outputs = self.encoder_decoder.encoder(batch['images'])

        # Samples:
        logits, sampled_token_ids, sample_str = self.sample(prompt['input_ids'], encoder_outputs)

        # Sample reward:
        labels = [[f'{i} {j}'] for i, j in zip(batch['findings'], batch['impression'])]
        reward = self.reward(sample_str, labels).to(self.device)  # batch contains the labels.

        # Baseline token identifiers:
        baseline_ids = self.encoder_decoder.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=prompt['input_ids'],
            special_token_ids=[
                self.tokenizer.additional_special_tokens_ids[
                    self.tokenizer.additional_special_tokens.index('[PMT-SEP]')
                ],
                self.tokenizer.bos_token_id,
                self.tokenizer.sep_token_id,
            ],            
            max_length=self.decoder_max_len + prompt['input_ids'].shape[1],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # An update to generate() now prepends bos_token_id to each sequence if it does not exist at the start of the input: 
        #   https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/generation/utils.py#L699C13-L699C30
        # Hence, we remove the prepended bos_token_id from each sequence if it is there:
        if torch.all(baseline_ids[:, 0] == 1):
            baseline_ids = baseline_ids[:, 1:]

        # Findings and impression sections (exclude previous impression section):
        _, baseline_findings, baseline_impression = self.encoder_decoder.split_and_decode_sections(
            baseline_ids,
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        baseline = self.reward(
            [f'{i} {j}' for i, j in zip(baseline_findings, baseline_impression)], labels,
        ).to(self.device)  # batch contains the labels.
        reward = reward - baseline

        # Loss:
        loss = self.reinforce_loss(logits, sampled_token_ids, reward)

        # Update and log scores for each metric:
        self.log_dict(
            {'reward': torch.mean(reward), 'baseline': torch.mean(baseline)},
            on_step=True,
            on_epoch=True,
            batch_size=batch['images'].size()[0],
        )

        return loss

    def sample(
        self,
        prompt_ids: torch.Tensor,
        encoder_outputs: transformers.modeling_outputs.BaseModelOutput,
    ):
        """
        Generate the sample caption for SCST.

        Argument/s:
            prompt_ids - token identifiers of the previous impression section to prompt the next report.
            encoder_outputs - encoder outputs for the decoder's cross-attention.

        Returns:
            logits - logits from the output of the language model head.
            sampled_token_ids - sampled token indices.
            sample_str - the sampled captions.
        """

        sample = self.encoder_decoder.generate.__wrapped__(  # Use __wrapped__ to avoid the torch.no_grad() decorator of generate().
            self.encoder_decoder,
            input_ids=prompt_ids.to(self.device),
            special_token_ids=[self.tokenizer.bos_token_id, self.tokenizer.sep_token_id],
            encoder_outputs=encoder_outputs,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            do_sample=True,
            num_beams=1,
            use_cache=True,
            output_scores=True,
            top_p=self.scst_sample_top_p,
            top_k=self.scst_sample_top_k,
            temperature=self.scst_sample_temperature,
            max_new_tokens=self.decoder_max_len - 1,  # BOS token is already included.
        )

        # An update to generate() now prepends bos_token_id to each sequence if it does not exist at the start of the input: 
        #   https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/generation/utils.py#L699C13-L699C30
        # Hence, we remove the prepended bos_token_id from each sequence if it is there:
        if torch.all(sample['sequences'][:, 0] == 1):
            sample['sequences'] = sample['sequences'][:, 1:]

        # Logits
        logits = torch.stack(sample['scores'], dim=-1)

        # Findings and impression sections (exclude previous impression section):
        _, findings, impression = self.encoder_decoder.split_and_decode_sections(
            sample['sequences'],
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        sample_str = [f'{i} {j}' for i, j in zip(findings, impression)]

        # Sampled token IDs
        sampled_token_ids = sample['sequences'][:, prompt_ids.shape[1]:]

        # Sequence length
        mask = sampled_token_ids == self.tokenizer.pad_token_id
        seq_len = torch.sum(torch.logical_not(mask), dim=-1).float()

        # Log sequence length
        self.log_dict({'seq_len': torch.mean(seq_len)}, on_step=True, on_epoch=True, batch_size=seq_len.size()[0])

        return logits, sampled_token_ids, sample_str

    def reinforce_loss(self, logits: torch.Tensor, sampled_token_ids: torch.Tensor,
                       reward: torch.Tensor) -> torch.Tensor:
        """
        Loss for the REINFORCE algorithm from https://doi.org/10.1007/BF00992696. It is detailed for
        gradient descent in https://doi.org/10.1109/cvpr.2017.131.

        PyTorch implementation:
            https://pytorch.org/docs/stable/distributions.html#score-function

        Argument/s
            logits - logits from the language model head.
            sampled_token_ids - sampled token indices.
            reward - reward for each batch element.

        Returns:
            REINFORCE loss for gradient descent.
        """

        # Negative log-likelihood of each sampled token
        loss = torch.nn.functional.nll_loss(
            input=torch.nn.functional.log_softmax(logits, dim=1),
            target=sampled_token_ids,
            ignore_index=self.tokenizer.pad_token_id,
            reduction='none',
        )

        # Negative sequence log-likelihood:
        loss = loss.sum(dim=-1)

        # Reward:
        loss = loss * reward

        # Mean over mini-batch elements:
        loss = loss.mean()

        return loss


class GTPromptCXRBERT(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = CXRBERTReward(device=self.device)
