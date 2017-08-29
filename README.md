# kaggle-quora-question-pairs
My solution to [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) competition (Top 7%, 209th of 3307).

## Running
`python quora_solver.py config/<config json file>`<br><br>
Parameters (data locations, network architecture, pre-trained word vectors, number of epochs, etc) can be specified in config json file (examples are in `config/`).

## Requirements
### Dataset
* [Quora question pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data)
### Pre-trained word vectors
* [Google word2vec](https://drive.google.com/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM)
* [GloVe](http://nlp.stanford.edu/data/glove.840B.300d.zip)
### Dependencies
Install Python dependencies using `pip install -r requirements.txt`.
