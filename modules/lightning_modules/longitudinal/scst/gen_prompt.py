import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader

from data.prompt import PreviousReportSubset
from modules.lightning_modules.longitudinal.gen_prompt import GeneratedPrompt
from tools.rewards.cxrbert import CXRBERTReward


class SCSTGeneratedPrompt(GeneratedPrompt):

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
        super(SCSTGeneratedPrompt, self).__init__(**kwargs)

        self.trial = trial
        self.scst_sample_top_p = scst_sample_top_p
        self.scst_sample_top_k = scst_sample_top_k
        self.scst_sample_temperature = scst_sample_temperature

        assert self.mbatch_size == 1

        # Freeze the encoder:
        for p in self.encoder_decoder.encoder.parameters():
            p.requires_grad = False

        # Unfreeze all parameters of the decoder (even LoRA):
        for p in self.encoder_decoder.decoder.parameters():
            p.requires_grad = True

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        # Load the merged MIMIC-CXR .csv file:
        df = pd.read_csv(self.merged_csv_path)

        # Dataframe that provides the history for each study:
        history = df.copy()

        # Drop studies that don't have a findings or impression section:
        df = df.dropna(subset=['findings', 'impression'], how='any')

        # Drop studies that have more than the maximum number of DICOMs per study:
        df = df[df.study_id.map(df.study_id.value_counts()) <= self.max_images_per_study]

        if stage == 'fit' or stage is None:
            self.train_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'train'],
                history=history.loc[history['split'] == 'train'].copy(),
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.train_transforms,
                use_generated=True,
                scst_generated=True,
                mbatch_size=self.mbatch_size,
            )
            print(f'No. of training examples: {self.train_set.__len__()}.')
            print(
                f'No. of training dicom_ids & study_ids: {self.train_set.df.dicom_id.nunique()}',
                f'& {self.train_set.df.study_id.nunique()}.',
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'validate'],
                history=history.loc[history['split'] == 'validate'].copy(),
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                use_generated=True,
                mbatch_size=self.mbatch_size,
            )
            print(f'No. of validation examples: {self.val_set.__len__()}.')
            print(
                f'No. of validation dicom_ids & study_ids: {self.val_set.df.dicom_id.nunique()}',
                f'& {self.val_set.df.study_id.nunique()}.',
            )

        if stage == 'test' or stage is None:
            self.test_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'test'],
                history=history.loc[history['split'] == 'test'].copy(),
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                use_generated=True,
                mbatch_size=self.mbatch_size,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids & study_ids: {self.test_set.df.dicom_id.nunique()}',
                f'& {self.test_set.df.study_id.nunique()}.',
            )

    def train_dataloader(self):
        """
        Training set DataLoader.

        Returns:
            DataLoader.
        """
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_set, shuffle=False)
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            num_workers=0,
            collate_fn=self.collate_fn,
            shuffle=False,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=0,  # Only one worker as generated previous impression will be missed.
            shuffle=False,
            collate_fn=self.collate_fn,
        )  # Cannot use prefetch_factor as generated previous impression will be missed.

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=0,  # Only one worker as generated previous impression will be missed.
            shuffle=False,
            collate_fn=self.collate_fn,
        )  # Cannot use prefetch_factor as generated previous impression will be missed.

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        raise NotImplementedError

    def on_train_epoch_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-epoch-start
        """
        self.train_set.allocate_subjects_to_rank(seed=(self.current_epoch + self.trial + 1)*(self.trial + 1))
        self.train_set.df['generated_impression'] = np.nan
        self.val_set.df['generated_impression'] = np.nan

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

        # Add the generated impressions the dataframe of the validation set:
        for i, j, k in zip(batch['study_ids'], baseline_findings, baseline_impression):
            self.train_set.history.loc[self.train_set.history.study_id == i, 'generated_findings'] = j
            self.train_set.history.loc[self.train_set.history.study_id == i, 'generated_impression'] = k

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
            encoder_outputs - cross-attention module encoder inputs.

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


class GeneratedPromptCXRBERT(SCSTGeneratedPrompt):

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = CXRBERTReward(device=self.device)
