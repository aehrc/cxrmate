import torch
import transformers

from task.mimic_cxr.model.report_gen.any.variable import VariableCXR
from task.mimic_cxr.tools.rewards.cxrbert import CXRBERTReward
from task.mimic_cxr.tools.rewards.rewards import RadGraphReward


class SCSTVariable(VariableCXR):

    def __init__(
            self,
            scst_sample_top_p: float = 1.0,
            scst_sample_top_k: float = 50,
            scst_sample_temperature: float = 1.0,
            **kwargs,
    ):
        """
        Argument/s:
            scst_sample_top_p - only the most probable tokens with probabilities that add up to top_p or higher are
                considered during sampling.
            scst_sample_top_k - only the top-k ranked tokens are considered during sampling.
            scst_sample_temperature - the sharpness of the softmax probability distribution during sampling.
            kwargs - keyword arguments.
        """
        super(SCSTVariable, self).__init__(**kwargs)

        self.scst_sample_top_p = scst_sample_top_p
        self.scst_sample_top_k = scst_sample_top_k
        self.scst_sample_temperature = scst_sample_temperature

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # SCST step
        loss = self.scst_step(batch, batch_idx)

        # Logging
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric.
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

        # Encoder outputs:
        encoder_outputs, attention_mask = self.encoder_forward(batch['images'], batch['dicom_study_ids'])

        # Samples:
        logits, sampled_token_ids, sample_str = self.sample(encoder_outputs, attention_mask)

        # Sample reward:
        labels = [[f'{i[0]} {j[0]}'] for i, j in zip(batch['findings'], batch['impression'])]
        reward = self.reward(sample_str, labels).to(self.device)

        # Baseline token identifiers:
        baseline_ids = self.generate(            
            num_beams=1, 
            dicom_study_ids=batch['dicom_study_ids'], 
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )

        # Findings and impression sections (exclude previous impression section):
        baseline_findings, baseline_impression = self.split_and_decode_sections(
            baseline_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
        )
        baseline = self.reward(
            [f'{i} {j}' for i, j in zip(baseline_findings, baseline_impression)], labels,
        ).to(self.device)
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

    def sample(self, encoder_outputs: transformers.modeling_outputs.BaseModelOutput, attention_mask: torch.Tensor):
        """
        Generate the sample caption for SCST.

        Argument/s:
            encoder_outputs - cross-attention module encoder inputs.
            attention_mask - cross-attention keys mask.

        Returns:
            logits - logits from the output of the language model head.
            sampled_token_ids - sampled token indices.
            sample_str - the sampled captions.
        """

        # BOS token identifiers:
        bos_ids = torch.ones(
            (encoder_outputs.last_hidden_state.size()[0], 1), dtype=torch.long, device=self.device
        ) * self.tokenizer.bos_token_id

        # Logits warper:
        logits_warper = transformers.LogitsProcessorList(
            [
                transformers.TemperatureLogitsWarper(self.scst_sample_temperature),
                transformers.TopPLogitsWarper(self.scst_sample_top_p),
                transformers.TopKLogitsWarper(self.scst_sample_top_k),
            ]
        )

        # Stopping criteria:
        stopping_criteria = transformers.StoppingCriteriaList([
            transformers.MaxLengthCriteria(max_length=self.decoder_max_len)],
        )

        sample = self.encoder_decoder.sample(
            special_token_ids=[self.tokenizer.sep_token_id],
            input_ids=bos_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            do_sample=True,
            use_cache=True,
            output_scores=True,
        )

        # Logits
        logits = torch.stack(sample['scores'], dim=-1)

        # Convert token indices into strings for the reward function
        findings, impression = self.split_and_decode_sections(
            sample['sequences'],
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
        )
        sample_str = [f'{i} {j}' for i, j in zip(findings, impression)]

        # Sampled token IDs
        sampled_token_ids = sample['sequences'][:, 1:]

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

        # Negative sequence log-likelihood
        loss = loss.sum(dim=-1)

        # Reward
        loss = loss * reward

        # Mean over mini-batch elements
        loss = loss.mean()

        return loss
    
class CXRBERT(SCSTVariable):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = CXRBERTReward(ckpt_dir=self.ckpt_zoo_dir, device=self.device)

   
class RGER(SCSTVariable):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = RadGraphReward(
            radgraph_reward='rg_er',
            module_load_apptainer=self.module_load_apptainer,
            image_dir=self.image_dir,
            device=self.device,
        )
    