import os
from argparse import Namespace

import torch
import transformers
from dlhpcstarter.trainer import trainer_instance
from dlhpcstarter.utils import (
    get_test_ckpt_path,
    importer,
    load_config_and_update_args,
    resume_from_ckpt_path,
    write_test_ckpt_path,
)
from lightning.pytorch import seed_everything


def stages(args: Namespace):
    """
    Handles the training and testing stages for the task. This is the stages() function
        defined in the task's stages.py.

    Argument/s:
        args - an object containing the configuration for the job.
    """
    args.warm_start_modules = False

    # Set seed number (using the trial number) for deterministic training
    seed_everything(args.trial, workers=True)

    # Get configuration & update args attributes
    # Note: this needs to be called again for submitted jobs due to partial parsing.
    load_config_and_update_args(args)

    # If AMD Instinct GPU:
    if torch.cuda.is_available():
        if torch.cuda.get_device_name() == 'AMD Radeon Graphics':
            torch.set_float32_matmul_precision('medium')

    # Model definition:
    TaskModel = importer(definition=args.definition, module=args.module)

    # Trainer:
    trainer = trainer_instance(**vars(args))

    # Train:
    if args.train:

        # For deterministic training: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Warm-start from checkpoint
        if args.warm_start_ckpt_path:
            model = TaskModel.load_from_checkpoint(checkpoint_path=args.warm_start_ckpt_path, **vars(args))
            print('Warm-starting using: {}.'.format(args.warm_start_ckpt_path))

        # Warm-start from other experiment:
        elif hasattr(args, 'warm_start_exp_dir'):
            if args.warm_start_exp_dir:
                
                assert isinstance(args.warm_start_exp_dir, str)

                # The experiment trial directory of the other configuration:
                warm_start_exp_dir_trial = os.path.join(args.warm_start_exp_dir, f'trial_{args.trial}')

                # Get the path to the best performing checkpoint
                ckpt_path = get_test_ckpt_path(
                    warm_start_exp_dir_trial, 
                    args.warm_start_monitor, 
                    args.warm_start_monitor_mode, 
                    args.test_epoch, 
                    args.test_ckpt_path,
                )

                model = TaskModel.load_from_checkpoint(checkpoint_path=ckpt_path, **vars(args))
                print('Warm-starting using: {}.'.format(ckpt_path))

        # Warm-start from Hugging Face checkpoint:
        elif hasattr(args, 'warm_start_name'):
            if args.warm_start_name:
                model = TaskModel(**vars(args))
                hf_ckpt = transformers.AutoModel.from_pretrained(args.warm_start_name, trust_remote_code=True)
                model.encoder_decoder.load_state_dict(hf_ckpt.state_dict())
        else:
            args.warm_start_modules = True
            model = TaskModel(**vars(args))

        # Train
        ckpt_path = resume_from_ckpt_path(args.exp_dir_trial, args.resume_last, args.resume_epoch, args.resume_ckpt_path)
        trainer.fit(model, ckpt_path=ckpt_path)

    # Test
    if args.test:
        if hasattr(args, 'test_ckpt_name') and 'model' not in locals():
            model = TaskModel(**vars(args))
            hf_ckpt = transformers.AutoModel.from_pretrained(args.test_ckpt_name, trust_remote_code=True)
            model.encoder_decoder.load_state_dict(hf_ckpt.state_dict())
        elif args.fast_dev_run:
            if 'model' not in locals():
                model = TaskModel(**vars(args))
        else:

            # Get the path to the best performing checkpoint
            ckpt_path = get_test_ckpt_path(
                args.exp_dir_trial, args.monitor, args.monitor_mode, args.test_epoch, args.test_ckpt_path,
            )

            print('Testing checkpoint: {}.'.format(ckpt_path))
            write_test_ckpt_path(ckpt_path, args.exp_dir_trial)

            model = TaskModel.load_from_checkpoint(checkpoint_path=ckpt_path, **vars(args))

        trainer.test(model)
