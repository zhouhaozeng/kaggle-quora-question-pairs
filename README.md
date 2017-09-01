# kaggle-quora-question-pairs
My solution to [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) competition (Top 7%, 209th of 3307).

## Overview

My solution is a deep learning architecture.
* Questions are firstly pre-processed (lowercase, remove stop words, expand shortened words, etc) and encoded as padded sequences.
* Question sequences are mapped to vectors by pre-trained word embeddings. The encoded question vectors are then fed into a LSTM layer.
* Meanwhile, statistical features (question occurrence and co-occurrence counts) are extracted from raw data and fed into a dense (fully connected) layer.
* The two LSTM layer outputs (for question pairs) and dense layer output (for statistical features) are concatenated and fed into a dense layer to produce the final classification result.

Below is my solution diagram.
<br><br>

<img src="images/solution_diagram.png" width="900">

## Running
`python quora_solver.py config/<config json file>`

Parameters (data locations, network architecture, pre-trained word vectors, number of epochs, etc) can be specified in config json file (examples are in `config/`).

Both prediction (`submission_*.csv`) and trained model (`model_*.h5`) will be saved.

## Requirements
### Dataset
* [Quora question pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data)
### Pre-trained word vectors
* [Google word2vec](https://drive.google.com/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM)
* [GloVe](http://nlp.stanford.edu/data/glove.840B.300d.zip)
### Dependencies
Install Python dependencies using `pip install -r requirements.txt`.
