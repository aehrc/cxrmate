# cxrmate-single-tf:
sshpass -p $SETONIX_PASSWORD ssh anicolson@data-mover.pawsey.org.au "mkdir -p /scratch/pawsey0864/anicolson/experiments/mimic_cxr/082_any_single/trial_1/"
sshpass -p $SETONIX_PASSWORD scp /datasets/work/hb-mlaifsp-mm/work/experiments/mimic_cxr/082_any_single/trial_1/epoch=11-val_report_chexbert_f1_macro=0.347756.ckpt \
    anicolson@data-mover.pawsey.org.au:/scratch/pawsey0864/anicolson/experiments/mimic_cxr/082_any_single/trial_1/

# cxrmate-variable-tf:
sshpass -p $SETONIX_PASSWORD ssh anicolson@data-mover.pawsey.org.au "mkdir -p /scratch/pawsey0864/anicolson/experiments/mimic_cxr/083_any_variable/trial_1/"
sshpass -p $SETONIX_PASSWORD scp /datasets/work/hb-mlaifsp-mm/work/experiments/mimic_cxr/083_any_variable/trial_1/epoch=28-val_report_chexbert_f1_macro=0.383505.ckpt \
    anicolson@data-mover.pawsey.org.au:/scratch/pawsey0864/anicolson/experiments/mimic_cxr/083_any_variable/trial_1/

# cxrmate-tf:
sshpass -p $SETONIX_PASSWORD ssh anicolson@data-mover.pawsey.org.au "mkdir -p /scratch/pawsey0864/anicolson/experiments/mimic_cxr/091_any_prompt_variable_lora/trial_0/"
sshpass -p $SETONIX_PASSWORD scp /datasets/work/hb-mlaifsp-mm/work/experiments/mimic_cxr/091_any_prompt_variable_lora/trial_0/epoch=6-step=27433-val_report_chexbert_f1_macro=0.388249.ckpt \
    anicolson@data-mover.pawsey.org.au:/scratch/pawsey0864/anicolson/experiments/mimic_cxr/091_any_prompt_variable_lora/trial_0/

# cxrmate:
sshpass -p $SETONIX_PASSWORD ssh anicolson@data-mover.pawsey.org.au "mkdir -p /scratch/pawsey0864/anicolson/experiments/mimic_cxr/098_gen_prompt_cxr_bert/trial_0/"
sshpass -p $SETONIX_PASSWORD scp /datasets/work/hb-mlaifsp-mm/work/experiments/mimic_cxr/098_gen_prompt_cxr_bert/trial_0/epoch=0-step=3917-val_report_chexbert_f1_macro=0.425015.ckpt \
    anicolson@data-mover.pawsey.org.au:/scratch/pawsey0864/anicolson/experiments/mimic_cxr/098_gen_prompt_cxr_bert/trial_0/
