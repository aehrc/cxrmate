import torch
import transformers

from task.mimic_cxr.model.report_gen.any.prompt_variable_lora import \
    GTPromptLoRA
from task.mimic_cxr.tools.rewards.ccr import ClinicallyCoherentReward
from task.mimic_cxr.tools.rewards.cider import COCOCIDErReward
from task.mimic_cxr.tools.rewards.cxrbert import CXRBERTReward
from task.mimic_cxr.tools.rewards.factent import (FactENTBERTScoreReward,
                                                  FactENTNLIBERTScoreReward,
                                                  FactENTNLIReward,
                                                  FactENTReward)
from task.mimic_cxr.tools.rewards.radgraph import RadGraphReward


class SCSTGTPrompt(GTPromptLoRA):

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
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder_projection.parameters():
            p.requires_grad = False
        for p in self.last_hidden_state_layer_norm.parameters():
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
        prompt = self.tokenize_prompt(
            batch['previous_findings'], batch['previous_impression'], add_bos_token_id=True,
        )

        # Encoder outputs:
        encoder_outputs, attention_mask = self.encoder_forward(batch['images'], batch['dicom_study_ids'])

        # Samples:
        logits, sampled_token_ids, sample_str = self.sample(prompt['input_ids'], encoder_outputs, attention_mask)

        # Sample reward:
        labels = [[f'{i} {j}'] for i, j in zip(batch['findings'], batch['impression'])]
        reward = self.reward(sample_str, labels).to(self.device)  # batch contains the labels.

        # Baseline token identifiers:
        baseline_ids = self.generate(
            num_beams=1, 
            dicom_study_ids=batch['dicom_study_ids'], 
            prompt_ids=prompt['input_ids'],
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )

        # Findings and impression sections (exclude previous impression section):
        _, baseline_findings, baseline_impression = self.split_and_decode_sections(
            baseline_ids,
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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
        attention_mask: torch.Tensor,
    ):
        """
        Generate the sample caption for SCST.

        Argument/s:
            prompt_ids - token identifiers of the previous impression section to prompt the next report.
            encoder_outputs - cross-attention module encoder inputs.
            attention_mask - cross-attention keys mask.

        Returns:
            logits - logits from the output of the language model head.
            sampled_token_ids - sampled token indices.
            sample_str - the sampled captions.
        """

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
            transformers.MaxLengthCriteria(max_length=self.decoder_max_len + prompt_ids.shape[1])],
        )

        sample = self.encoder_decoder.sample(
            input_ids=prompt_ids.to(self.device),
            special_token_ids=[self.tokenizer.bos_token_id, self.tokenizer.sep_token_id],
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            prompt_token_id=self.tokenizer.additional_special_tokens_ids[
                self.tokenizer.additional_special_tokens.index('[PMT]')
            ],
            return_dict_in_generate=True,
            do_sample=True,
            use_cache=True,
            output_scores=True,
        )

        # Logits
        logits = torch.stack(sample['scores'], dim=-1)

        # Findings and impression sections (exclude previous impression section):
        _, findings, impression = self.split_and_decode_sections(
            sample['sequences'],
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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
        self.reward = CXRBERTReward(ckpt_dir=self.ckpt_zoo_dir, device=self.device)


class GTPromptRGERHat(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = RadGraphReward(
            radgraph_reward='rg_er_hat',
            module_load_apptainer=self.module_load_apptainer,
            image_dir=self.image_dir,
            device=self.device,
        )


class GTPromptRGER(SCSTGTPrompt):

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


class GTPromptRGE(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = RadGraphReward(
            radgraph_reward='rg_e',
            module_load_apptainer=self.module_load_apptainer,
            image_dir=self.image_dir,
            device=self.device,
        )


class GTPromptFactEnt(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        self.reward = FactENTReward()


class GTPromptFactEntBERTScore(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        self.reward = FactENTBERTScoreReward(
            ckpt_dir=self.ckpt_zoo_dir, device=self.device, num_workers=self.num_workers,
        )


class GTPromptFactENTNLI(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        self.reward = FactENTNLIReward(
            mbatch_size=self.fact_entnli_mbatch_size,
            num_workers=self.num_workers,
            module_load_apptainer=self.module_load_apptainer,
            image_dir=self.image_dir,
            device=self.device,
        )


class GTPromptFactENTNLIBERTScore(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        self.reward = FactENTNLIBERTScoreReward(
            mbatch_size=self.fact_entnli_mbatch_size,
            num_workers=self.num_workers,
            module_load_apptainer=self.module_load_apptainer,
            image_dir=self.image_dir,
            device=self.device,
            ckpt_dir=self.ckpt_zoo_dir,
        )


class GTPromptCCR(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        self.reward = ClinicallyCoherentReward(
            device=self.device,
            ckpt_dir=self.ckpt_zoo_dir,
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
        )


class GTPromptCIDEr(SCSTGTPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        # Reward for SCST is initialised here as the dataframe is not available in self.__init__()
        self.reward = COCOCIDErReward(labels=(self.train_set.df.findings + ' ' + self.train_set.df.impression).tolist())
