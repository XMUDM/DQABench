<div align='center'>
    <h1>Question Classification Model</h1>
</div>

## Contents

* [Overview](#overview)
* [Requirements](#overview)
* [Setup](#setup)
* [Data Preparation](#data-preparation)
* [Files](#files)
* [Download Link](#download-link)
* [Statistics of the Dataset](#statistics-of-the-dataset)

## Overview

 In addition to using the large model for autonomous classification, we implemented two additional classification methods. This section contains the specific implementation of two methods based on the XLNet transformer.

## Requirements

- Python 3.9
- PyTorch 1.12.0+cu113
- Transformers 4.32.1
- scikit-learn 1.3.1
- pandas 2.0.3
- tqdm 4.66.1
- Matplotlib

## Setup

1. Install the required packages:
    ```bash
    pip install torch transformers scikit-learn pandas matplotlib
    ```

2. Download the pretrained XLNet model from HuggingFace:
    - [Chinese XLNet](https://huggingface.co/hfl/chinese-xlnet-base)
    - [English XLNet](https://huggingface.co/xlnet/xlnet-base-cased)

## Data Preparation

Ensure your dataset is in CSV format with the following structure:

| label | text          |
|-------|---------------|
| 0     | sample text 1 |
| 1     | sample text 2 |

The CSV file should not have a header.

## Files

1. **train.py**: Script to train the model.
2. **eval.py**: Script to evaluate the trained model.
3. **prepare_data.py**: Contains functions to load and preprocess the data.
4. **multi_steps.py**: The implementation of our multi_step classification approach.
5. **one_step.py**: The implementation of our one_step classification approach.
6. **checkpoints**: 
   1. ***db_en_multi_ckpt.tar:*** The checkpoint of English database related classifier, with labels: \[general, gauss, tool, other\].
   2. ***db_zh_multi_ckpt.tar:*** The checkpoint of Chinese database related classifier, with labels: \[general, gauss, tool, other\].
   3. ***safety_en_ckpt.tar:*** The checkpoint of English safety related classifier, with labels: \[unsafe, other\].
   4. ***safety_zh_ckpt.tar:*** The checkpoint of Chinese safety related classifier, with labels: \[unsafe, other\].
   5. ***en_5class.tar:*** The checkpoint of the English five-category classifier, with labels: \[general, gauss, tool, other, unsafe\].
   6. ***zh_5class.tar:*** The checkpoint of the Chinese five-category classifier, with labels: \[general, gauss, tool, other, unsafe\].

### Training

#### Command-line Arguments

- `--pretrained_model`: Directory of the pretrained model
- `--train_dataset`: Path to the training dataset
- `--val_dataset`: Path to the validation dataset
- `--save_dir`: Directory to save the model checkpoints
- `--imgs_dir`: Directory to save the training and validation plots
- `--last_new_checkpoint`: Path to the last checkpoint
- `--labels`: List of labels
- `--LR`: Learning rate
- `--EPOCHS`: Number of epochs
- `--batch_size`: Batch size
- `--doc_maxlen`: Maximum input length
- `--segment_len`: Segment length
- `--overlap`: Length of overlap between segments
- `--ngpu`: Number of GPUs
- `--feature_extract`: Boolean, whether to use XLNet as a feature extractor
- `--train_from_scratch`: Boolean, whether to start training from scratch

#### Usage

```sh
python train.py \
    --pretrained_model /path/to/pretrained/model/ \
    --train_dataset /path/to/train/dataset.csv \
    --val_dataset /path/to/val/dataset.csv \
    --save_dir /path/to/save/dir/ \
    --imgs_dir /path/to/imgs/dir/ \
    --labels ['unsafe', 'other'] \
    --LR 5e-4 \
    --EPOCHS 10 \
    --batch_size 512 \
    --doc_maxlen 4000 \
    --segment_len 256 \
    --overlap 50 \
    --ngpu 1 \
    --feature_extract True \
    --train_from_scratch True
```

### Evaluation

#### Command-line Arguments

- `--pretrained_model`: Directory of the pretrained model
- `--test_dataset`: Path to the test dataset
- `--save_dir`: Directory where the model checkpoints are saved
- `--last_new_checkpoint`: Path to the last checkpoint
- `--labels`: List of labels
- `--batch_size`: Batch size
- `--doc_maxlen`: Maximum input length
- `--segment_len`: Segment length
- `--ngpu`: Number of GPUs
- `--feature_extract`: Boolean, whether to use XLNet as a feature extractor

#### Usage

```sh
python eval.py \
    --pretrained_model /path/to/pretrained/model/ \
    --test_dataset /path/to/test/dataset.csv \
    --save_dir /path/to/save/dir/ \
    --last_new_checkpoint /path/to/last/checkpoint \
    --labels ['unsafe', 'other'] \
    --batch_size 128 \
    --doc_maxlen 4000 \
    --segment_len 256 \
    --ngpu 1 \
    --feature_extract True
```

### Multi-step Evaluation

The `multi_steps_eval.py` script is designed for evaluating models on both English and Chinese text datasets. It includes safety and database related classification, using separate models for each language and task.

#### Command-line Arguments

- `--safety_model_en`: Path to the English safety classifier checkpoint.
- `--safety_model_zh`: Path to the Chinese safety classifier checkpoint.
- `--db_model_en`: Path to the English database related classifier checkpoint.
- `--db_model_zh`: Path to the Chinese database related classifier checkpoint.
- `--doc_maxlen`: Maximum input length.
- `--segment_len`: Segment length.
- `--ngpu`: Number of GPUs.
- `--pretrained_model_en`: Directory of the English pretrained model (e.g., download from https://huggingface.co/xlnet/xlnet-base-cased).
- `--pretrained_model_zh`: Directory of the Chinese pretrained model (e.g., download from https://huggingface.co/hfl/chinese-xlnet-base).
- `--test_set`: Path to the test dataset.

#### Usage

```sh
python multi_steps_eval.py \
    --safety_model_en ckpt/safety_en_ckpt.tar \
    --safety_model_zh ckpt/safety_zh_ckpt.tar \
    --db_model_en ckpt/db_en_multi_ckpt.tar \
    --db_model_zh ckpt/db_zh_multi_ckpt.tar \
    --doc_maxlen 4000 \
    --segment_len 256 \
    --ngpu 1 \
    --pretrained_model_en /path/to/english/pretrained/model/ \
    --pretrained_model_zh /path/to/chinese/pretrained/model/ \
    --test_set /path/to/test/dataset.csv
```

### One-Step Evaluation

The `one_step_eval.py` script is designed to perform a comprehensive evaluation of both English and Chinese text classification models. It loads the models, processes the test dataset, and computes evaluation metrics. Below is an explanation of the script and its usage.

#### Command-line Arguments

- `--model_en`: Path to the English classifier checkpoint.
- `--model_zh`: Path to the Chinese classifier checkpoint.
- `--doc_maxlen`: Maximum input length.
- `--segment_len`: Segment length.
- `--overlap`: Length of overlap between segments.
- `--ngpu`: Number of GPUs.
- `--pretrained_model_en`: Directory of the English pretrained model (e.g., download from https://huggingface.co/xlnet/xlnet-base-cased).
- `--pretrained_model_zh`: Directory of the Chinese pretrained model (e.g., download from https://huggingface.co/hfl/chinese-xlnet-base).
- `--test_set`: Path to the test set.

#### Usage

```sh
python one_step_eval.py \
    --model_en /path/to/english/checkpoint \
    --model_zh /path/to/chinese/checkpoint \
    --doc_maxlen 4000 \
    --segment_len 256 \
    --overlap 50 \
    --ngpu 1 \
    --pretrained_model_en /path/to/english/pretrained/model/ \
    --pretrained_model_zh /path/to/chinese/pretrained/model/ \
    --test_set /path/to/test/dataset.csv
```

## Download Link

### Test Set

Please download the complete test set for classifiers at [test set](https://drive.google.com/file/d/19KekFUssISzHNsFStt90OGGQBx2j0SwK/view?usp=drive_link).

### Checkpoints

Please download the checkpoints of classifiers at [checkpoints](https://drive.google.com/file/d/1nHYF0vo6rZlstBPObop0s-WGh3qKRaAK/view?usp=drive_link).

## Statistics of the Dataset

<table>
    <tr>
        <td>Classifier</td>
        <td>Language</td>
        <td>Dataset Type</td>
        <td>Sourse</td>
        <td>Training set #N</td>
        <td>Validation set #N</td>
        <td>Test set #N</td>
    </tr>
    <tr>
        <td rowspan="4">security classifier</td>
        <td rowspan="2">ZH</td>
        <td>Unsafe questions</td>
        <td>Safety-Prompts</td>
        <td>61478</td>
        <td>8000</td>
        <td></td>
    </tr>
    <tr>
        <td>Safe questions</td>
        <td>Alpaca, Longbench, questions from DQA</td>
        <td>66049</td>
        <td>8000</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">EN</td>
        <td>Unsafe questions</td>
        <td>Generated with Safety-Prompts method</td>
        <td>55958</td>
        <td>8000</td>
        <td></td>
    </tr>
    <tr>
        <td>Safe questions</td>
        <td>Alpaca, Longbench, questions from DQA</td>
        <td>66045</td>
        <td>8000</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="8">database classifier</td>
        <td rowspan="4">ZH</td>
        <td>general</td>
        <td>General Knowledge questions from DQA</td>
        <td>60000</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td>gauss</td>
        <td>Database Product questions from DQA</td>
        <td>29520</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td>tool</td>
        <td>Database Instance questions from DQA</td>
        <td>20000</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td>irrelevant</td>
        <td>Alpaca, Longbench</td>
        <td>51130</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">EN</td>
        <td>general</td>
        <td>General Knowledge questions from DQA</td>
        <td>60000</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td>gauss</td>
        <td>Database Product questions from DQA</td>
        <td>29520</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td>tool</td>
        <td>Database Instance questions from DQA</td>
        <td>20000</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td>irrelevant</td>
        <td>Alpaca, Longbench</td>
        <td>52550</td>
        <td>114</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="10">Naïve calssifier</td>
        <td rowspan="5">ZH</td>
        <td>Unsafe questions</td>
        <td>Safety-Prompts; Translated from BeaverTails-Evaluation(only for test)</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>general</td>
        <td>General Knowledge questions from DQA</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>gauss</td>
        <td>Database Product questions from DQA</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>tool</td>
        <td>Database Instance questions from DQA</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>irrelevant</td>
        <td>Alpaca, Longbench</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td rowspan="5">EN</td>
        <td>Unsafe questions</td>
        <td>Generated with Safety-Prompts method; BeaverTails-Evaluation(only for test)</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>general</td>
        <td>General Knowledge questions from DQA</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>gauss</td>
        <td>Database Product questions from DQA</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>tool</td>
        <td>Database Instance questions from DQA</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>
    <tr>
        <td>irrelevant</td>
        <td>Alpaca, Longbench</td>
        <td>20000</td>
        <td>500</td>
        <td>500</td>
    </tr>

Note: All questions in alpaca and longbench are filtered based on database keyword matching to ensure that there are no database-related questions.
