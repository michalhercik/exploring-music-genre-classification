# Music genre classification

## Abstract

In this work I put together few scripts to extract features from dataset, train
three neural network models (MLP, CNN, LSTM) and predict genres for given files.
It was my first experience with music analysis and as the main source of
information I used Valerio Velardos
[series](https://youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&si=TJ67v4J0N-aiQoKJ)
about deep learning for audio. In next sections I will briefly describe dataset,
each script, network architectures, accuracies and at the end I've written down
some final thoughts.

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
data. Then from each segment Mel Frequency Cepstrum Coefficients (MFCCs) are
extracted. MFCCs are derived from short-time Fourier transform (STFT) and
approximates human auditory system, which should be very handy in classifying
social construct such as music genre.

Since the dataset contains 100 audio files the preprocessed data has a following
dimensions *(1000, 13)* *TODO: check it!!!*.

## Neural network models

### MLP

The first layer (input layer) flattens the data because MLP expects single
dimension and preprocessed features are two dimensional *TODO: fact checking*.
MLP has three hidden layers with ReLU as an activation function. To deal with
overfitting each hidden layer uses L2 regularization and dropout. The output
layer uses softmax as an activation function.

### CNN

*TODO: input dimensions*

CNN has three convolution layers with max pooling and batch normalization, one
dense layer with dropout and dense output layer. Every layer has ReLU as an
activation function except output layer which has softmax.

### LSTM

LSTM (Long Short-Term Memory) has two LSTM layers then dense layer with dropout
and ReLU as an activation function and as a last a dense output layer with
softmax.

### Test Accuracy

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
talking about *91 %* accuracy on the same dataset using ensembles of ML
algorithms.

I tried some testing with random data from the i

## Final thoughts

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





