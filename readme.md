# Music genre classification

## Abstract

In this work I put together a few scripts to extract features from a dataset,
train a three neural network models (MLP, CNN, LSTM) and predict genres for a
given files. It was my first experience with music analysis and as the main
source of an information I used Valerio Velardos
[series](https://youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&si=TJ67v4J0N-aiQoKJ)
about deep learning for audio. In the following sections I will show how to
train the models, predict genres of an audio files using provided scripts and
briefly describe the dataset, each script, network architectures and its
performance.

## How to train and predict

The following steps describe how to train the models and predict genres for an
audio files. It is assumed that you have downloaded and extracted [GTZAN
Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
next to the scripts.

1. Extract features from the dataset and save them to a JSON file by running
   `preprocess.py`.
2. Train a model on the extracted features and save it by running one of the `mlp.py`,
   `cnn.py`, `lstm.py`.
3. Classify an audio files with `predict.py` script. This script can take the following arguments:
    - `--model <path_to_model>`: specify a path to the trained model.
    - `--input <path_to_audio>`: specify a path to a **folder** with the audio files (tested with `.mp3` and `.wav`).
    - `--type <min/max/mean>`: specify a type of an aggregation function for segments of an audio file (`mean` is default). 
    - `--count <1-10>`: specify a number of printed genre results (sorted, highest first). Default behavior is printed all unsorted. 

## Dataset

For training, validation and testing I have used audio files from [GTZAN
Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
It is a collection of *10* genres with *100* audio files each, all having a
length of a *30* seconds. The genres are blues, classical, country, disco,
hiphop, jazz, metal, pop, reggae and rock. Even though it is a very popular
dataset it has many flaws like mislabeling or duplicates [[*1*](#1),[*2*](#)].
Since trying methods for audio analysis is the main purpose of this project the
dataset quality is not that important aspect as a ease to use the data. For more
accurate performance results the FMA dataset [[*3*](#3)] would be a better fit,
especially the almost one terabyte large version. 

## Data preprocessing

Here I will not describe the exact parameters and steps used for the data
preprocessing rather I will give a high-level explanation. More accurate
description of the data preprocessing can be found in the scripts
`preprocess.py` which does the whole feature extraction and `config.py` which
defines a global variables used in the other scripts as well.

The first step is splitting each audio into ten segments to get more training
data. Then extract from each segment Mel Frequency Cepstrum Coefficients
(MFCCs). MFCC is derived from Fourier transformation (FT) and approximates human
auditory system, which should be very handy in classifying social construct such
as music genre. The final shape of preprocessed data is *(9986, 130, 13)*.
*9986* we get by splitting each audio sample into 10 segments and remove
corrupted segments. *130* is number of FT for each segment and *13* is number of
MFCCs.  

## Neural network models

As in the [Data preprocessing](#data-preprocessing) section I will gave a high
level description of the neural network architectures used for classification.

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
need to be extended with depth dimension so that it satisfies CNN input dimensions
requirements. 

### LSTM

LSTM (Long Short-Term Memory) has two LSTM layers then dense layer with dropout
and ReLU as an activation function and as a last a dense output layer with
softmax.

## Predictions

Prediction has a three phases - preprocessing, predicting and aggregating.
Preprocessing consists of loading first *30s* of an audio file (this implies the
minimal length of an audio is *30s*), then splitting it into *10* segments and
extracting features (MFCCs) as described in [Data
preprocessing](#data-preprocessing). Predicting is then done using chosen neural
network model. Lastly aggregating predicted genres of segments from the same
audio file either by choosing the minimal/maximal value or calculating the mean.
There is not really a reason for using first *30s* of an audio file, using the
whole audio file could yield a better results but it would be harder to
aggregate because of a variable number of segments for an audio file.

## Performance

Model | Test Accuracy (%)
------|------------------
CNN   | 73,9             
LSTM  | 64,2             
MLP   | 56,4             

This paper says [[*4*](#4)] that according to this paper [[*5*](#5)] a human
accuracy in a genre classification is on average *70%* for *10* genres. Since
the original paper is old, I couldn't verify it so let's just take it as a hard
to trust reference value and work with it. 

Surprisingly LSTM which should work great with a time series of features has not
the best results but this can be just because of a bad parameters. CNN is the
only one that surpassed the human accuracy. A similar accuracy is achieved in
this paper from *2018* [[*6*](#6)] but currently there are also papers talking
about much higher accuracies raising over *90%* [[*7*](#7), [*8*](#8),
[*9*](#9)]. It would almost sound that the problem is solved but this benchmarks
are done on GTZAN dataset which has as I mentioned in section
[Dataset](#dataset) many flaws. Neural networks benchmarked on the FMA dataset
are getting much worse results [[*10*](#10)].

I have also tried some testing with data downloaded from
[Pixabay](https://pixabay.com/music/). I searched for each genre on the website
and picked one audio file that sounded as the genre I searched for. The
prediction results can be seen in the table below displaying name of the audio,
expected genre, and predicted probability of each genre.

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

The results are not surprising, *6/10* is predicted correctly. It also looks
like some genres are harder to predict than others that can be also observed in
this paper [[*10*](#10)].

## Conclusion

Genre is not a well defined construct so its classification is a hard task. For
an example disco was pop music in 70s so maybe it would be better to do a music
genre classification with respect to time. Another issue with a genre is that
each genre contains sub-genres which is either completely new style or
combination of existing one from another genres so it can get very confusing and
new sub-genres are still emerging. Even though many new and better neural
network architectures are created it is not enough to correctly classify music
genre since it is not clear what that exactly means and even two people of the
same age don't have to agree on a genre simply because genre classification is
subjective.

Despite all that the current state of the art approaches are getting better
results than humans and with audio data growth hence active research in the
field is done, we could expect machines getting only better at it. To make this
project a useful one I would need to get better understanding of genres, audio
analysis, change dataset, improve neural network architecture and also focus
more on predicting but the main goal of this project was achieved - I learned a
lot about audio analysis and machine learning.

## Miscellaneous

Script `config.py` contains paths and constants definitions used in other
scripts. Feel free to change the values but remember all the scripts expects
that the defined paths already exists except the file itself. Script
`predictArgParser.py` is for parsing arguments passed to `predict.py` script.

## References

 1. <span id="1">An analysis of the GTZAN music genre dataset, [DOI](https://doi.org/10.1145/2390848.2390851)</span>
 2. <span id="2">The State of the Art Ten Years After a State of the Art: Future Research in Music Information Retrieval, [DOI](https://doi.org/10.1080/09298215.2014.894533)</span>
 3. <span id="3">FMA: A Dataset For Music Analysis, [DOI](https://doi.org/10.48550/arXiv.1612.01840)</span>
 4. <span id="4">Convolutional Neural Network Achieves Human-level Accuracy in Music Genre Classification, [DOI](https://doi.org/10.48550/arXiv.1802.09697)</span>
 5. <span id="5">Human accuracy [DOI](https://doi.org/10.1109/ICASSP.2004.1326806)</span>
 6. <span id="6">A Novel Music Genre Classification Using Convolutional Neural Network, [DOI](https://doi.org/10.1109/ICCES51350.2021.9489022)</span>
 7. <span id="7">Comparing Recurrent Neural Network Types in a Music Genre Classification Task: Gated Recurrent Unit Superiority Using the GTZAN Dataset, [DOI](https://www.researchgate.net/profile/Eric-Odle/publication/374698715_Comparing_Recurrent_Neural_Network_Types_in_a_Music_Genre_Classification_Task_Gated_Recurrent_Unit_Superiority_Using_the_GTZAN_Dataset/links/6529d3e81a05311a23fbe815/Comparing-Recurrent-Neural-Network-Types-in-a-Music-Genre-Classification-Task-Gated-Recurrent-Unit-Superiority-Using-the-GTZAN-Dataset.pdf)</span>
 8. <span id="8">Genre Classification in Music using Convolutional Neural Networks, [DOI](https://doi.org/10.1007/978-981-99-7339-2_33)</span>
 9. <span id="9">A Hybrid Model for Music Genre Classification Using LSTM and SVM [DOI](https://doi.org/10.1109/IC3.2018.8530557)</span>
 10. <span id="10">Music Genre Classification: Looking for the Perfect Network, [DOI](https://doi.org/10.1007/978-3-030-77961-0_6)</span>







