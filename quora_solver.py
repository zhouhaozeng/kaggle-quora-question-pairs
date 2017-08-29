__author__ = 'Zhouhao Zeng: https://www.kaggle.com/zhouhao'
__date__ = '6/12/2017'

import os
import re
import logging
import csv
import json
import time
import codecs
import numpy as np
import pandas as pd
import collections

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Reshape, Flatten, LSTM, Bidirectional, ConvLSTM2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def elapsed_time(start_time, end_time):
    elapsed_sec = end_time - start_time
    h = int(elapsed_sec / (60 * 60))
    m = int((elapsed_sec % (60 * 60)) / 60)
    s = int(elapsed_sec % 60)
    return "{}:{:>02}:{:>02}".format(h, m, s)


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)   


def generate_embedding_matrix(word_index, word2vec, embedding_dim):
    # generate embedding matrix from word vectors
    logging.info('Preparing embedding matrix')
    nb_words = len(word_index) + 1        
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    logging.info('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))    
    return embedding_matrix


# load GloVe word vectors
class glove_word2vec(object):
    def __init__(self, embedding_file):
        self.vocab = {}
        with open(embedding_file, 'rb') as f:
            for line in f:
                sline = line.split()
                self.vocab[sline[0]] = map(float, sline[1:])
    def word_vec(self, word):
        return self.vocab[word]


class Solver(object):   
    def __init__(self, config_file):
        self.name = re.search('(\S+)\.json', os.path.basename(config_file)).group(1)
        if not os.path.isdir(self.name):
            os.mkdir(self.name)
        self.output_dir = self.name + '/'
        
        self.timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

        # load parameters from config json file
        with open(config_file, 'r') as f:
            self.params = json.load(f) 

        # random initialization seed
        self.seed = self.params.get('seed', np.random.random_integers(100))

        # for hyper-parameters specified as "random" in config file, assign random value.
        logging.info('Assign random value for hyper-parameters if necessary')
        if self.params.get('num_lstm', None) == "random":
            self.params['num_lstm'] = np.random.randint(175, 275)
            print("num_lstm: %d" % self.params['num_lstm'])
        if self.params.get('num_dense', None) == "random":
            self.params['num_dense'] = np.random.randint(100, 150)
            print("num_dense: %d" % self.params['num_dense'])
        if self.params.get('rate_drop_lstm', None) == "random":
            self.params['rate_drop_lstm'] = 0.15 + np.random.rand() * 0.25
            print("rate_drop_lstm: %f" % self.params['rate_drop_lstm'])
        if self.params.get('rate_drop_dense', None) == "random":
            self.params['rate_drop_dense'] = 0.15 + np.random.rand() * 0.25   
            print("rate_drop_dense: %f" % self.params['rate_drop_dense'])  


    def load_word2vec(self):
        # read word vectors with from
        logging.info('Load word embeddings')
        
        if self.params['embedding_file_type'] == 'word2vec':
            self.word2vec = KeyedVectors.load_word2vec_format(self.params['embedding_file'], binary=True)
        elif self.params['embedding_file_type'] == 'glove':            
            self.word2vec = glove_word2vec(self.params['embedding_file'])
                
        logging.info('Found %s word vectors of word2vec' % len(self.word2vec.vocab))


    def load_data(self):
        # process texts in datasets
        texts_1, texts_2, labels, ids = self.load_train_data(self.params['train_data_file'])
        logging.info('Found %s texts in train data' % len(texts_1))
    
        test_texts_1, test_texts_2, test_ids = self.load_test_data(self.params['test_data_file'])
        logging.info('Found %s texts in test data' % len(test_texts_1)) 
    
        tokenizer = Tokenizer(num_words=self.params['max_nb_words'])
        tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)        
    
        sequences_1 = tokenizer.texts_to_sequences(texts_1)
        sequences_2 = tokenizer.texts_to_sequences(texts_2)
        test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
        test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
    
        word_index = tokenizer.word_index
        logging.info('Found %s unique tokens' % len(word_index))
    
        data_1 = pad_sequences(sequences_1, maxlen=self.params['max_seq_len'])
        data_2 = pad_sequences(sequences_2, maxlen=self.params['max_seq_len'])
        labels = np.array(labels)
        logging.info('Shape of data tensor: {}'.format(data_1.shape))
        logging.info('Shape of label tensor: {}'.format(labels.shape)) 

        feats, test_feats = self.build_features()
        logging.info('Built %s additional features' % self.feats_dim)
        
        train_data, val_data = self.train_val_split(data_1, data_2, feats, labels)
    
        test_data_1 = pad_sequences(test_sequences_1, maxlen=self.params['max_seq_len'])
        test_data_2 = pad_sequences(test_sequences_2, maxlen=self.params['max_seq_len'])
        test_ids = np.array(test_ids)
         
        test_data = {'X': [np.vstack((test_data_1, test_data_2)), np.vstack((test_data_2, test_data_1)), np.vstack((test_feats, test_feats))],
                     'ids': np.concatenate((test_ids, test_ids)),
                     }
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.word_index = word_index

    @staticmethod
    def load_train_data(train_data_file):
        texts_1 = [] 
        texts_2 = []
        labels = []
        ids = []
        with codecs.open(train_data_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for values in reader:
                texts_1.append(text_to_wordlist(values[3]))
                texts_2.append(text_to_wordlist(values[4]))
                labels.append(int(values[5]))
                ids.append(values[0])
        return texts_1, texts_2, labels, ids

    @staticmethod
    def load_test_data(test_data_file):
        test_texts_1 = []
        test_texts_2 = []
        test_ids = []
        with codecs.open(test_data_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for values in reader:
                test_texts_1.append(text_to_wordlist(values[1]))
                test_texts_2.append(text_to_wordlist(values[2]))
                test_ids.append(values[0])
        return test_texts_1, test_texts_2, test_ids   

    def build_features(self):
        # In addition to text sequence features, build additional statistical features into the model
        train_df = pd.read_csv(self.params['train_data_file'])
        test_df = pd.read_csv(self.params['test_data_file']) 
        
        ques = pd.concat([train_df[['question1', 'question2']], \
                          test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
        q_dict = collections.defaultdict(set)
        for i in range(ques.shape[0]):
            q_dict[ques.question1[i]].add(ques.question2[i])
            q_dict[ques.question2[i]].add(ques.question1[i])
                 
        def q1_freq(row):
            return(len(q_dict[row['question1']]))
        
        def q2_freq(row):
            return(len(q_dict[row['question2']]))
        
        def q1_q2_intersect(row):
            return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))) 
        
        train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
        train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
        train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)
        
        test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
        test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
        test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)
        
        feats = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
        test_feats = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
        
        ss = StandardScaler()
        ss.fit(np.vstack((feats, test_feats)))
        feats = ss.transform(feats)
        test_feats = ss.transform(test_feats)
        
        self.feats_dim = feats.shape[1]
        
        return feats, test_feats

    def train_val_split(self, data_1, data_2, feats, labels, validation_split=0.1):
        logging.info('Random seed for train validaiton data split: %s' % self.seed)
        np.random.seed(self.seed)
        perm = np.random.permutation(len(data_1))
        idx_train = perm[:int(len(data_1)*(1-validation_split))]
        idx_val = perm[int(len(data_1)*(1-validation_split)):]
        
        data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
        data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
        feats_train = np.vstack((feats[idx_train], feats[idx_train]))
        labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
        
        data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
        data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
        feats_val = np.vstack((feats[idx_val], feats[idx_val]))
        labels_val = np.concatenate((labels[idx_val], labels[idx_val]))
        
        weight_val = np.ones(len(labels_val))
        if self.params['re_weight']:
            weight_val *= 0.472001959
            weight_val[labels_val==0] = 1.309028344 
            
        train_data = {'X': [data_1_train, data_2_train, feats_train],
                      'Y': labels_train
                      }    
        
        val_data = {'X': [data_1_val, data_2_val, feats_val],
                    'Y': labels_val,
                    'weight': weight_val
                    }
            
        return train_data, val_data

    def load_embedding(self):
        # prepare word embeddings
        self.nb_words = len(self.word_index) + 1 
        self.embedding_matrix = generate_embedding_matrix(self.word_index, self.word2vec, self.params['embedding_dim'])  

    def load_model(self):
        if self.params['model'] == 'LSTM':
            embedding_layer = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'])
            lstm_layer = LSTM(self.params['num_lstm'], dropout=self.params['rate_drop_lstm'], recurrent_dropout=self.params['rate_drop_lstm'])
            
            sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1 = embedding_layer(sequence_1_input)
            x1 = lstm_layer(embedded_sequences_1)
            
            sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2 = embedding_layer(sequence_2_input)
            y1 = lstm_layer(embedded_sequences_2)
            
            feats_input = Input(shape=(self.feats_dim,))
            feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)                    
            
            merged = concatenate([x1, y1, feats_dense])
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            merged = Dense(self.params['num_dense'], activation=self.params['dense_activation'])(merged)
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            preds = Dense(1, activation='sigmoid')(merged)
            
            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
            model.compile(loss='binary_crossentropy',
                          optimizer='nadam',
                          metrics=['acc']) 
            
        elif self.params['model'] == "bidirectional_LSTM":
            embedding_layer = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'])
            lstm_layer = Bidirectional(LSTM(self.params['num_lstm'], dropout=self.params['rate_drop_lstm'], recurrent_dropout=self.params['rate_drop_lstm']))
            
            sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1 = embedding_layer(sequence_1_input)
            x1 = lstm_layer(embedded_sequences_1)
            
            sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2 = embedding_layer(sequence_2_input)
            y1 = lstm_layer(embedded_sequences_2)
            
            feats_input = Input(shape=(self.feats_dim,))
            feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)                
            
            merged = concatenate([x1, y1, feats_dense])
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            merged = Dense(self.params['num_dense'], activation=self.params['dense_activation'])(merged)
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            preds = Dense(1, activation='sigmoid')(merged)
            
            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
            model.compile(loss='binary_crossentropy',
                          optimizer='nadam',
                          metrics=['acc'])
            
        elif self.params['model'] == 'ConvLSTM':
            embedding_layer = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'])     
            
            sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1 = embedding_layer(sequence_1_input)
            
            sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2 = embedding_layer(sequence_2_input)
            
            num_filters = 20
            kernel_size = 50
            stride_size = 10
            ConvLSTM_layer = ConvLSTM2D(filters=num_filters, kernel_size=(kernel_size, 2), input_shape=(None, self.params['embedding_dim'], 2, 1), strides=(stride_size, 1), padding='same', \
                                        dropout=self.params['rate_drop_lstm'], recurrent_dropout=self.params['rate_drop_lstm'], return_sequences=False)
            
            sequence_merged = concatenate([embedded_sequences_1, embedded_sequences_2])
            sequence_merged = Reshape((-1, self.params['embedding_dim'], 2, 1))(sequence_merged)
            sequence_ConvLSTM = ConvLSTM_layer(sequence_merged)
            sequence_ConvLSTM = BatchNormalization()(sequence_ConvLSTM)
            sequence_ConvLSTM_dim = np.product(sequence_ConvLSTM.shape.as_list()[1:])
            sequence_ConvLSTM = Flatten()(sequence_ConvLSTM)
            sequence_ConvLSTM.set_shape((None, sequence_ConvLSTM_dim))            
            
            feats_input = Input(shape=(self.feats_dim,))
            feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)                    
            
            merged = concatenate([sequence_ConvLSTM, feats_dense])
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            merged = Dense(self.params['num_dense'], activation=self.params['dense_activation'])(merged)
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            preds = Dense(1, activation='sigmoid')(merged)
            
            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
            model.compile(loss='binary_crossentropy',
                          optimizer='nadam',
                          metrics=['acc'])
           
        self.model = model
    
    def preprocess(self):
        self.load_word2vec()
        self.load_data()
        self.load_embedding()
        self.load_model()       
        
    def train(self):
        if self.params['re_weight']:
            class_weight = {0: 1.309028344, 1: 0.472001959}
        else:
            class_weight = None        

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        best_model_fname = self.output_dir + 'model_' + self.name + '_%s'%(self.timestamp) + '.h5'
        model_checkpoint = ModelCheckpoint(best_model_fname, save_best_only=True, save_weights_only=True)
        
        hist = self.model.fit(self.train_data['X'], self.train_data['Y'], \
                              validation_data=(self.val_data['X'], self.val_data['Y'], self.val_data['weight']), \
                              epochs=self.params['nb_epoches'], batch_size=2048, shuffle=True, verbose=2, \
                              class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
        
        self.model.load_weights(best_model_fname)
        self.best_val_score = min(hist.history['val_loss'])  
        
    def predict(self):
        # make the submission
        logging.info('Start making the submission')
        
        preds = self.model.predict(self.test_data['X'], batch_size=2048, verbose=2)
        preds = preds.reshape((2,-1)).mean(axis=0)
        test_ids = self.test_data['ids'].reshape((2,-1))[0,:]
        
        submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
        submission.to_csv(self.output_dir + 'submission_' + self.name + '_%.4f'%(self.best_val_score) + '_%s'%(self.timestamp) + '.csv', index=False)


def main():
    start_time = time.time()
    
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s]: %(message)s ',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        filemode="w"
                        )        
    
    config_file = sys.argv[1]
    quora_solver = Solver(config_file)

    quora_solver.preprocess()    
    quora_solver.train()
    quora_solver.predict()
     
    end_time = time.time()
    logging.info("Run complete: %s elapsed" % elapsed_time(start_time, end_time))

    
if __name__ == '__main__':
    main()