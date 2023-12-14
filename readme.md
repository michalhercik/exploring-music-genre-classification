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
[GTZANDataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
It is a collection of 10 genres with 100 audio files each, all having a length
30 seconds. The genres are blues, classical, country, disco, hiphop, jazz,
metal, pop, reggae, rock.

## Feature extraction

## Neural network models

### MLP

### CNN

### LSTM

## Prediction

## Comparing accuracy

CNN     73.9 %
LSTM    64.2 %
MLP     56.4 %

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





