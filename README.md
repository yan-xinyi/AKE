# Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs

## Overview
Data and source Code for the paper "Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs".

Nowadays, Automatic Keyphrase Extraction (AKE) with single eye-tracking source is constrained by physiological mechanism, signal processing techniques and other factors. In this paper, we propose to utilize EEG and eye-tracking signals to enhance AKE from Microblogs. Our work includes the followig aspects:

  - We applied different types of cognitive signals generated during human reading to AKE from Microblogs for the first time. Specifically, we combine EEG signals and Eye-tracking signals jointly to AKE based on the open-source cognitive language processing corpus [ZUCO](https://www.nature.com/articles/sdata2018291).
  -  We compared the effects of different frequency bands of EEG signals on the performance of the AKE.
  -  Furthermore, we evaluated AKE by combining the most effective EEG signals and eye-tracking signals from single-source cognitive signal tests.
  -  Analyzing the unsatisfactory results of the previous experiments, we improved the AKE model based on Pretrained Language Models (PLMs): First, we incorporate Glove embeddings into the input layer of the SATT-BiLSTM+CRF model, which exhibited the best AKE test performance. Second, we propose an improved AKE based on BERT. Lastly, we implemented an improved AKE based on the T5 (including T5-Base and T5-Large).

The results verified the enhancement of cognitive signals genarated during human reading on AKE. EEG signals exhibit the most significant improvement, while the combined results showed no further enhancement. T5-Large model can maximize the performance of the model without weakening the cognitive signals’ weights.

## Directory structure
```Root Directory
AKE
├── dataset
│   ├── Zuco
│   │    ├── test
│   │    └── train
│   ├── Election-Trec
│   │    ├── test
│   │    └── train
│   └── General-Twitter
│        ├── test
│        └── train
├── models
│   ├── pretrain_pt
│   │    ├── bert.pt
│   │    └── t5.pt
│   ├── BILSTM.py
│   ├── ATT-BILSTM.py
│   ├── ATT-BILSTM.py
│   ├── ATT-BILSTM+CRF.py
│   ├── SATT-BILSTM.py
│   ├── SATT-BILSTM+CRF.py
│   ├── SATT-BILSTM+CRF+GloVe.py
│   ├── BERT.ipynb
│   └── T5.ipynb
├── result
│   ├── Election-Trec
│   └── General-Twitter
├── config.py
├── utils.py
├── evaluate.py
├── processing.py
└── main.py
```

## Dataset discription
We release our all train and test data in "dataset" directory, In the dataset below, cognitive features have been spliced between each word and the corresponding label:
1. Election-Trec Dataset: The Election-Trec dataset4 is derived from the open-source dataset TREC2011 track4. After removing all "#" symbols, it contains 24,210 training tweets and 6,054 testing tweets.
2. General-Twitter Dataset: Developed by (Zhang et al., 2016), employs Hashtags as keyphrases for each tweet. It consists of 78,760 training tweets and 33,755 testing tweets, with an average sentence length of about 13 words.

Meaning of each column of dataset:
- Column 1: words
- Columns 2 to 18: word-level eye-tracking signals
- Columns 19 to 26: word-level EEG signals
- Column 27: labels

Meaning of each row of the data:
- Empty lines indicate a sentence break, and one consecutive paragraph represents a sentence.

## Requirements
First, our system environment is set up according to the following configuration:
- Python==3.7
- Torch==1.8.0
- torchvision==0.9.0
- Sklearn==0.0
- Numpy 1.25.1+mkl
- nltk==3.6.2
- Tqdm==4.56.0

## Quick start
### Implementation steps for Bi-LSTM based experiments:
1. Run the processing.py file to process the data into json format:
    `python processing.py`
2. Configure hyperparameters in the `config.py` file. There are roughly the following parameters to set:
    - `modeltype`: select which model to use for training and testing.
    - `train_path`,`test_path`,`vocab_path`,`save_path`: path of train data, test data, vocab data and results.
    - `fs_name`, `fs_num`: Name and number of cognitive traits.
    - `run_times`: Number of repetitions of training and testing.
    - `epochs`: refers to the number of times the entire training dataset is passed through the model during the training process. 
    - `lr`: learning rate.
    - `vocab_size`: the size of vocabulary. 37347 for Election-Trec Dataset, 85535 for General-Twitter.
    - `embed_dim`,`hidden_dim`: dim of embedding layer and hidden layer.
    - `batch_size`: refers to the number of examples (or samples) that are processed together in a single forward/backward pass during the training or inference process of a machine learning model.
    - `max_length`: is a parameter that specifies the maximum length (number of tokens) allowed for a sequence of text input. It is often used in natural language processing tasks, such as text generation or text classification.
3. Modifying combinations of additive cognitive features in the model. For example, the code below means add all 25 features into the model:
    `input = torch.cat([input, inputs['et'], inputs['eeg']], dim=-1)`
4. based on your system, open the terminal in the root directory 'AKE' and type this command:
    `python main.py` 

### Implementation steps for Large Language Models(LLMs) based experiments:
1. BERT: Run `BERT.ipynb` in the `models/` directory:
    - Run the code in top-to-bottom order. 
    - Cognitive signals added in the model construction: `outputs = torch.concat((bert_outputs,extra_features[:,:,:]),-1)`.
    - Set epoch to 5 and train the model. Save the model parameter with the best F1 value to the path under `models/pretrain_pt`.
    - When testing, the model parameters are read from `models/pretrain_pt`.
2. T5-Base: Run `T5.ipynb` in the `models/` directory:
    - Set parameter weight = 't5-base'.
    - Cognitive signals added in the model construction: `outputs = torch.concat((T5_outputs,extra_features[:,:,:]),-1)`. 
    - Set epoch to 5 and train the model. Save the model parameter with the best F1 value to the path under `models/pretrain_pt`.
    - When testing, the model parameters are read from `models/pretrain_pt`.
3. T5-Large: Run `T5.ipynb` in the `models/` directory:
    - Unlike t5-Base, set parameter weight = 't5-large'.
    - Other steps are similar to the above.
  
## Citation
Please cite the following paper if you use these codes and datasets in your work.

> Xinyi Yan, Yingyi Zhang, Chengzhi Zhang. Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs. ***Information Management & Processing***, 2023 (under review). 
