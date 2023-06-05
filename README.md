# CXRMate: Longitudinal Chest X-Ray Report Generation

|![](docs/tokens.drawio.png)|
|----|
| <p align="center"> <a>CXRMate: longitudinal, variable-CXR report generation. The decoder is prompted by the impression section of the previous study. [PMT], [BOS],  [SEP], and [EOS] denote the *prompt*, *beginning-of-sentence*, *separator*, and *end-of-sentence* special tokens, respectively.</a> </p> |

# Generated reports:
<!-- Generated reports for the single-CXR, variable-CXR, and longitudinal, variable-CXR (both prompted with the ground truth and the generated reports) are located in the [`generated_reports`](https://github.com/aehrc/cxrmate/blob/main/generated_reports) directory. -->
Generated reports for the single-CXR, variable-CXR, and longitudinal, variable-CXR (both prompted with the ground truth and the generated reports) are located in the [`generated_reports`](https://anonymous.4open.science/r/cxrmate-D1D3/generated_reports) directory.

## Installation:
After cloning the repository, install the required packages in a virtual environment.
The required packages are located in `requirements.txt`:
```shell script
python -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt --no-cache-dir
```

# Hugging Face model:

Singe-CXR: https://huggingface.co/aehrc/mimic-cxr-report-gen-single

## MIMIC-CXR Dataset:   

 - The MIMIC-CXR-JPG dataset is available at: 
        ```
        https://physionet.org/content/mimic-cxr-jpg/2.0.0/
        ```

## Run testing:   

The model configurations for each task can be found in its `config` directory, e.g. `config/test_mimic_cxr_chen_cvt2distilgpt2.yaml`. To run testing:

```shell
dlhpcstarter -t mimic_cxr_chen -c config/test_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --test
```

See [`dlhpcstarter==0.1.2`](https://github.com/csiro-mlai/dl_hpc_starter_pack) for more options. 

Note: data will be saved in the experiment directory (`exp_dir` in the configuration file).

## Run training:
   
To train with teacher forcing:
 
```
dlhpcstarter -t mimic_cxr -c config/train_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --train --test
```

To then train with Self-Critical Sequence Training (SCST) with the CXR-BERT reward:

 1. Copy the path to the teacher forcing checkpoint to the configuration file for SCST.
 2. 
    ```
    dlhpcstarter -t mimic_cxr -c config/train_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --train --test
    ```

See [`dlhpcstarter==0.1.2`](https://github.com/csiro-mlai/dl_hpc_starter_pack) for more options. 

## Help
If you need help, please leave an issue and we will get back to you as soon as possible.


