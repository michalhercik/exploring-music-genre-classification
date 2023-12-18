# Music genre classification

## Abstract

In this work I put together few scripts to extract features from dataset, train
three neural network models (MLP, CNN, LSTM) and predict genres for given files.
It was my first experience with music analysis and as the main source of
information I used Valerio Velardos
[series](https://youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&si=TJ67v4J0N-aiQoKJ)
about deep learning for audio. In next sections I will show how to train models,
predict genres of audio files and briefly describe dataset, each script, network
architectures and accuracies.

## How to train and predict

The following steps describe how to train models and predict genres for audio
files. It is assumed that you have downloaded and extracted [GTZAN
Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
next to the scripts.
1. Extract features from dataset and save them to JSON file by running
   `preprocess.py`.
2. Train model on extracted features and save it by running one of `mlp.py`,
   `cnn.py`, `lstm.py`.
3. Classify audio files with `predict.py` script. This script can take few arguments:
    - `--model <path>`:
    - `--input <path>`:
    - `--type <mean/max>`:
    - `--count <1-10>`: 





## Dataset

For training, validation and testing I used audio files from 
[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
It is a collection of 10 genres with 100 audio files each, all having a length
30 seconds. The genres are blues, classical, country, disco, hiphop, jazz,
metal, pop, reggae, rock.

Even though it is a very popular dataset it has many flaws like mislabeling or
duplicates *TODO: reference*. Since trying methods for audio analysis is the
main purpose of this project the dataset quality is not that important aspect as
ease to use the data. For more accurate performance results the FMA dataset
*TODO: reference* would be a better fit. 

## Data preprocessing

Here I will not describe the exact parameters and steps used for data
preprocessing rather I will give a high-level explanation. More accurate
description of data preprocessing can be found in scripts `preprocess.py` which
does the whole feature extraction and `config.py` which defines a global
variables used in other scripts as well.

The first step is splitting each audio into ten segments to get more training
data. Then extract from each segment Mel Frequency Cepstrum Coefficients
(MFCCs). MFCC is derived from Fourier transformation (FT) and approximates human
auditory system, which should be very handy in classifying social construct such
as music genre. The final shape of preprocessed data is *(9986, 130, 13)*.
*9986* we get by splitting each audio sample into 10 segments and remove
corrupted segments. *130* is number of FT for each segment and 13 is number of
MFCCs.  

## Neural network models

### MLP

The first layer (input layer) flattens the data because MLP expects two
dimensions and preprocessed features are three dimensional. MLP has three hidden
layers with ReLU as an activation function. To deal with overfitting each hidden
layer uses L2 regularization and dropout. The output layer uses softmax as an
activation function.

### CNN

CNN has three convolution layers with max pooling and batch normalization, one
dense layer with dropout and dense output layer. Every layer has ReLU as an
activation function except output layer which has softmax. Preprocessed data
need to be extend with depth dimension so that it satisfies CNN input dimensions
requirements. 

### LSTM

LSTM (Long Short-Term Memory) has two LSTM layers then dense layer with dropout
and ReLU as an activation function and as a last a dense output layer with
softmax.

### Preformance

Model | Test Accuracy (%)
------|------------------
CNN   | 73,9             
LSTM  | 64,2             
MLP   | 56,4             

This paper says *TODO: reference* that according to this paper *TODO: reference*
a human accuracy in genre classification is on average *70 %* for *10* genres.
Since the original paper is old, I couldn't verify it so let's just take it as a
hard to trust reference value and work with it. 

CNN is the only one that beated the human accuracy and according to some papers
*TODO: reference* the accuracy is not bad overall but there are also papers
talking about *91%* accuracy *TODO: reference* on the same dataset using
ensembles of ML algorithms but chasing higher accuracies on a dataset that is
not a good representation of a real world is a questionable goal.

I have also tried some testing with random data downloaded from
[Pixabay](https://pixabay.com/music/) *TODO: describe* and the result were
shockingly bad. From each downloaded audio is used only first *30s* and it is
preprocessed as described in [Data preprocessing](#data-preprocessing) section.
To make prediction audio segments of the same file are combined together either
by averaging or choosing maximal value. The next table shows prediction results
for each genre.

| Name | Genre | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|------|-------|-------|-----------|---------|-------|--------|------|-------|-----|--------|------|
Slow piano blues | Blues | **0.14** | 0.99 | 0.15 | 0.07 | 0.12 | 0.38 | 0.02 | 0.40 | 0.06 | 0.13 |
For elise prelude beethoven classic grand piano music | Classical | 0.33 | **0.98** | 0.27 | 0.02 | 0.29 | 0.49 | 0.00 | 0.02 | 0.04 | 0.01 |
Cowboy country | Country | 0.06 | 0.00 | **0.13** | 0.46 | 0.85 | 0.03 | 0.00 | 0.40 | 0.47 | 0.63 |
Disco groove | Disco | 0.99 | 0.00 | 0.03 | **0.16** | 0.98 | 0.01 | 0.02 | 0.07 | 0.10 | 0.02 |
Good night | Hip hop | 0.01 | 0.03 | 0.13 | 0.01 | **0.98** | 0.00 | 0.00 | 0.66 | 0.39 | 0.01 |
The best jazz club in new orleans | Jazz | 0.67 | 0.00 | 0.75 | 0.14 | 0.06 | **0.09** | 0.04 | 0.81 | 0.57 | 0.57 |
Frantic | Metal | 0.09 | 0.06 | 0.22 | 0.08 | 0.13 | 0.24 | **0.88** | 0.05 | 0.23 | 0.69 |
Abstract fashion pop | Pop | 0.39 | 0.26 | 0.03 | 0.42 | 0.99 | 0.07 | 0.01 | **0.80** | 0.42 | 0.06 |
Reggae island fun | Reggae | 0.00 | 0.00 | 0.02 | 0.99 | 0.11 | 0.00 | 0.00 | 0.73 | **0.99** | 0.04 |
Hard rock | Rock | 0.01 | 0.06 | 0.15 | 0.63 | 0.09 | 0.55 | 0.04 | 0.21 | 0.93 | **0.85** |


## References

### GTZAN has flaws -> mislabeling, duplicates

The State of the Art Ten Years After a State of the Art: Future Research in Music Information Retrieval
 - https://doi.org/10.1080/09298215.2014.894533

An analysis of the GTZAN music genre dataset
 - https://doi.org/10.1145/2390848.2390851

FMA: A Dataset For Music Analysis
 - https://doi.org/10.48550/arXiv.1612.01840

### Result is pleasing at first glance but not really (GTZAN flaws, and some testing)

Convolutional Neural Network Achieves Human-level Accuracy in Music Genre Classification
- https://doi.org/10.48550/arXiv.1802.09697

Human accuracy
 - https://doi.org/10.1109/ICASSP.2004.1326806

A Hybrid Model for Music Genre Classification Using LSTM and SVM
 - https://doi.org/10.1109/IC3.2018.8530557
 - interesting accuracy

A Novel Music Genre Classification Using Convolutional Neural Network
  - https://doi.org/10.1109/ICCES51350.2021.9489022

### Multilabel make more sense, pop is stupid

Music Genre Classification: Looking for the Perfect Network
 - https://doi.org/10.1007/978-3-030-77961-0_6
 - different dataset
 - conclusion - multilabel, pop is stupid

### Better neural networks architecture exists

Comparing Recurrent Neural Network Types in a Music Genre Classification Task: Gated Recurrent Unit Superiority Using the GTZAN Dataset
 - https://www.researchgate.net/profile/Eric-Odle/publication/374698715_Comparing_Recurrent_Neural_Network_Types_in_a_Music_Genre_Classification_Task_Gated_Recurrent_Unit_Superiority_Using_the_GTZAN_Dataset/links/6529d3e81a05311a23fbe815/Comparing-Recurrent-Neural-Network-Types-in-a-Music-Genre-Classification-Task-Gated-Recurrent-Unit-Superiority-Using-the-GTZAN-Dataset.pdf





