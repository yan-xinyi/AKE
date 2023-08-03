# Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs


## Overview
<b>Data and source Code for the paper "Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs".</b>

Nowadays, <b>Automatic Keyphrase Extraction (AKE)</b> with single eye-tracking source is constrained by physiological mechanism, signal processing techniques and other factors. 
<b>In this paper, we propose to utilize EEG and eye-tracking signals to enhance AKE from Microblogs. </b>
Our work includes the followig aspects:

  - We applied different types of cognitive signals generated during human reading to AKE from Microblogs for the first time. 
  Specifically, we combine EEG signals and Eye-tracking signals jointly to AKE based on the open-source cognitive language processing corpus ZUCO. 
  - We compared the effects of <b>different frequency bands of EEG signals</b>  on the performance of the AKE. 
  - Furthermore, we evaluated AKE by combining the most effective EEG signals and eye-tracking signals from single-source cognitive signal tests<  .
  - Analyzing the unsatisfactory results of the previous experiments, we improved the AKE model based on Pretrained Language Models (PLMs): First, we incorporate Glove embeddings into the input layer of the SATT-BiLSTM+CRF model, which exhibited the best AKE test performance. Second, we propose an improved AKE based on BERT. Lastly, we implemented an improved AKE based on the T5 (including T5-Base and T5-Large). 

The results verified the enhancement of cognitive signals genarated during human reading on AKE. EEG signals exhibit the most significant improvement, while the combined results showed no further enhancement. T5-Large model can maximize the performance of the model without weakening the cognitive signals’ weights.

## Directory Structure
<pre>AKE                                          Root directory
├── dataset                                  Experimental datasets
│   ├── ZUCO                                 Cognitive datasets
│   │    ├── test
│   │    └── train
│   └── Microblogs                           Microblogs based AKE datasets
│        ├── Election-Trec                   Election-Trec AKE Dataset
│        │     ├── test
│        │     └── train
│        └── General-Twitter                 General-Twitter AKE Dataset                 
│              ├── test
│              └── train
├── models                                   Module of the deep learning models and pre-trained models
│   ├── pretrain_pt                          Path to store pre-trained model parameters
│   │    ├── bert.pt
│   │    └── t5.pt
│   ├── BILSTM.py                            Baseline model
│   ├── ATT-BILSTM.py                        soft attention based Bi-LSTM
│   ├── SATT-BILSTM.py                       self-attention based Bi-LSTM
│   ├── ATT-BILSTM+CRF.py                    soft attention based Bi-LSTM+CRF
│   ├── SATT-BILSTM+CRF.py                   self-attention based Bi-LSTM+CRF
│   ├── SATT-BILSTM+CRF+GloVe.py             Improved model with GloVe Embeddings
│   ├── BERT.ipynb                           Improved model based on BERT model
│   └── T5.ipynb                             Improved model based on T5 model 
├── result                                   Path to store the results
│   ├── Election-Trec
│   └── General-Twitter
├── config.py                                Path configuration file
├── utils.py                                 Some auxiliary functions
├── evaluate.py                              Surce code for result evaluation
├── processing.py                            Source code of preprocessing function
├── main.py                                  Surce code for main function
└─README.md
</pre>

## Dataset Discription
In our study, two kinds of data are used: the cognitive signal data from human readings behaviors and the AKE from Microblogs data.
### 1. Cognitive Signal Data -- ZUCO Dataset
In this study, we choose <b>the Zurich Cognitive Language Processing Corpus ([ZUCO](https://www.nature.com/articles/sdata2018291))</b>, which captures eye-tracking signals and EEG signals of 12 adult native speakers reading approximately 1100 English sentences in normal and task reading modes. The raw data can be visited at: https://osf.io/2urht/#!. 

Only data from <b>the normal reading mode</b> were utilized to align with human natural reading habits. The reading corpus includes two datasets: 400 movie reviews from the Stanford Sentiment Treebank and 300 paragraphs about celebrities from the Wikipedia Relation Extraction Corpus. We release our all train and test data in “dataset” directory, In the ZUCO dataset, cognitive features have been spliced between each word and the corresponding label. 

Specifically, there are <b>17 Eye-tracking features</b> and <b>8 EEG features</b> were extracted from the dataset:

- <b>Eye-tracking features</b>
  In ZUCO Corpus, Hollenstein et al.(2019) categorized the 17 eye-tracking features into three groups(Refer to Table 1): Early-Stage Features,Late-Stage Features and Contextual Features, encompassing all gaze behavior stages and contextual influences.
    - Early-Stage Features reflect readers' initial comprehension and cognitive processing of the text.
    - Late-Stage Features indicate readers' syntactic and semantic comprehension.
    - Contextual Features refer to the gaze behavior of readers on the words surrounding the current word.


<div align=center>
Table 1. Summary of Eye-Tracking Features
<img src="https://yan-xinyi.github.io/figures/ET_features.png" width="750px" alt="Table 1. Summary of Eye-Tracking Features">
</div>

- <b>EEG features</b>
  EEG is a bio-electrical signal measurement used to assess brain activity by detecting electrical potential changes in brain neurons through multiple scalp electrodes. <b>Frequency domain analysis</b>, or spectral analysis, is a widely utilized EEG analysis method in various scientific disciplines. The recorded EEG signals used a 128-channel neural signal acquisition system, categorized into <b>four frequency bands</b> with two features per band (refer to Table 2 for details).


<div align=center> 
  
Table 2. Summary of EEG Features<br>
<img src="https://yan-xinyi.github.io/figures/EEG_features.png" width="450px" alt="Table 2. Summary of EEG Features">
</div>


### 2. AKE Dataset
- <b>Election-Trec Dataset</b>

  The Election-Trec dataset4 is derived from the open-source dataset TREC2011 track4. The raw data can be visited at: https://trec.nist.gov/data/tweets/. After removing all "#" symbols, it contains 24,210 training tweets and 6,054 testing tweets.

- <b>General-Twitter Dataset</b>

  Developed by (Zhang et al., 2016), employs Hashtags as keyphrases for each tweet. The raw data can be visited at: http://qizhang.info/paper/data/keyphrase_dataset.tar.gz. It consists of 78,760 training tweets and 33,755 testing tweets, with an average sentence length of about 13 words.
  Empty lines indicate a sentence break, and one consecutive paragraph represents a sentence.

## Requirements
System environment is set up according to the following configuration:
- Python==3.7
- Torch==1.8.0
- torchvision==0.9.0
- Sklearn==0.0
- Numpy 1.25.1+mkl
- nltk==3.6.2
- Tqdm==4.56.0

## Quick Start
### Implementation Steps for Bi-LSTM-based AKE
1. <b>Processing:</b> Run the processing.py file to process the data into json format:
    `python processing.py`

   The data is preprocessed to the format like: {['word','Value_et1',... ,'Value_et17','Value_eeg1',... ,'Value_eeg8','tag']}

2. <b>Configuration:</b> Configure hyperparameters in the `config.py` file. There are roughly the following parameters to set:
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
3. <b>Modeling:</b> Modifying combinations of additive cognitive features in the model.

   For example, the code below means add all 25 features into the model:

         `input = torch.cat([input, inputs['et'], inputs['eeg']], dim=-1)`
5. <b>Training and testing:</b> based on your system, open the terminal in the root directory 'AKE' and type this command:
    `python main.py` 

### Implementation Steps for Large Language Models(LLMs)-based AKE
1. <b>BERT:</b> Run `BERT.ipynb` in the `models/` directory:
     - Cognitive signals added in the model construction: `outputs = torch.concat((bert_outputs,extra_features[:,:,:]),-1)`.
     - Set epoch to 5 and train the model. Save the model parameter with the best F1 value to the path under `models/pretrain_pt`.
     - When testing, the model parameters are read from `models/pretrain_pt`.
2. <b>T5-Base:</b> Run `T5.ipynb` in the `models/` directory:
     - Set parameter weight = 't5-base'.
     - Cognitive signals are added in the `model construction` part: `outputs = torch.concat((T5_outputs,extra_features[:,:,:]),-1)`. 
     - Set epoch to 5 and train the model. Save the model parameter with the best F1 value to the path under `models/pretrain_pt`.
     - When testing, the model parameters are read from `models/pretrain_pt`.
3. <b>T5-Large:</b> Run `T5.ipynb` in the `models/` directory:
     - Unlike t5-Base, set parameter weight = 't5-large'.
     - Other steps are similar to the above.
  
## Case Study
We randomly selected five instances from the Election-Trec dataset and the General-Twitter dataset to visually illustrate the impact of cognitive signals generated during human reading on AKE from Microblogs (refer to Table 3 for details). 

In this study, we compared the performance of the AKE under four feature combinations:<b> "-," "EEG," "ET," and "ET&EEG"</b>. "-" indicates the model without using any cognitive processing signals. "EEG" and "ET" represent the model with only EEG signals and only eye-tracking signals, respectively. "ET&EEG" indicates the model that combines both eye-tracking and EEG signals simultaneously.

<div align=center> 
Table 3. Example of AKE incorporating Cognitive Signals Generated during Human Reading <br>
<img src="https://yan-xinyi.github.io/figures/Case Study.png" width="600px" alt="Table 3. Example of AKE incorporating Cognitive Signals Generated during Human Reading">


</div>
<b>Note</b>: Bold italicize mark indicates annotated correct Hashtags in microblog manually , blue mark represents predicted keyphrases correctly, green mark indicates predicted incorrect results, yellow mark represents partially predicted words for the target answers.
<br><br>



In order to compare the evaluation results more intuitively, we used the following scoring criteria: 10 points for correct predictions, 3 points for partially correct predictions, and 0 points for incorrect predictions. The scores for each feature combination were as follows:" - : 12 points, EEG : 86 points, ET : 29 points, and ET&EEG : 53 points". These results clearly indicate that cognitive signals generated during human reading have a positive impact on the AKE from Microblogs. Among them, EEG signals show a stronger enhancement on AKE performance, while eye-tracking signals exhibit a relatively weaker enhancing capability.


## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Xinyi Yan, Yingyi Zhang, Chengzhi Zhang. Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs. ***Information Processing and Management***, 2023 (Under Review).
