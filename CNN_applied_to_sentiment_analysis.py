#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob
import os
import pickle
from random import shuffle
import numpy as np

from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

word_vector = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',
       binary=True,
       limit=20000)

class CNN:
    def __init__(self, filepath):
        self.filepath = filepath
        self.maxlen = 400 # data to show the net before backpropagate and update weights
        self.batch_size = 32
        self.embedding_dims = 300 # Length of the token vectors to be passed to the convnet
        self.filters = 250 # N of filters to train
        self.kernel_size = 3 # with of the filters
        self.hidden_dims = 250 # numbers of neurons
        self.epochs = 2 # numer of times that the train data will be passed to the network

    def pre_process_data(self):
        positive_path = os.path.join(self.filepath, 'pos')
        negative_path = os.path.join(self.filepath, 'neg')
        pos_label = 1
        neg_label = 0
        dataset = []

        for filename in glob.glob(os.path.join(positive_path, '*.txt')):
            with open(filename, 'r') as f:
                dataset.append((pos_label, f.read()))

        for filename in glob.glob(os.path.join(negative_path, '*.txt')):
            with open(filename, 'r') as f:
                dataset.append((neg_label, f.read()))

        shuffle(dataset)

        return dataset

    def tokenize_and_vectorize(self, dataset):
        tokenizer = TreebankWordTokenizer()
        vectorized_data = []
        expected = []

        for sample in dataset:
            tokens = tokenizer.tokenize(sample[1])
            sample_vecs = []

            for token in tokens:
                try:
                    sample_vecs.append(word_vector[token])

                except KeyError:
                    pass

            vectorized_data.append(sample_vecs)

        return vectorized_data

    def collect_expected(self, dataset):
        # expected results
        expected = []

        for sample in dataset:
            expected.append(sample[0])

        return expected

    def save_data_as_file(self, filename, data):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        f.close()

        print('{} saved!'.format(filename))

    def load_file_as_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        f.close()

        return data

    def get_train_test_formatted(self, datas, label):
        split_point = int(len(datas) * .8)

        x_train = datas[:split_point]
        y_train = label[:split_point]
        x_test = datas[split_point:]
        y_test = label[split_point:]

        x_train = self.pad_truncate(x_train, self.maxlen)
        x_test = self.pad_truncate(x_test, self.maxlen)
        print(x_train)
        print(type(x_train))
        x_train = np.reshape(x_train, (len(x_train), self.maxlen, self.embedding_dims))
        y_train = np.array(y_train)
        x_test = np.reshape(x_test, (len(x_test), self.maxlen, self.embedding_dims))
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test

    def pad_truncate(self, data, maxlen):
        """
        The dimension of the datas to be shown to the convolution
        network must have the same dimension size.
        This function add ZERO pading to the input dimension less
        then the maxlen
        """
        new_data = []

        # Create a vector of 0s length of our word vectors
        zero_vector = []
        for _ in range(len(data[0][0])):
            zero_vector.append(0.0)

        for sample in data:
            if len(sample) > maxlen:
                temp = sample[:maxlen]
            elif len(sample) < maxlen:
                temp = sample
                additional_elems = maxlen - len(sample)

                for _ in range(additional_elems):
                    temp.append(zero_vector)
            else:
                temp = sample

            new_data.append(temp)

        return new_data

    def build_model(self):
        print('Build model...')

        model = Sequential([
            Conv1D(self.filters,
                self.kernel_size,
                padding='valid',
                activation='relu',
                strides=1,
                input_shape=(self.maxlen, self.embedding_dim))])

        return model


if __name__ == '__main__':

    filepath = 'aclImdb_v1/mdb/train'

    cnn = CNN(filepath)
    dataset = cnn.pre_process_data()

    vectorized_data = cnn.tokenize_and_vectorize(dataset)
    expected = cnn.collect_expected(dataset)

    x_train, y_train, x_test, y_test = cnn.get_train_test_formatted(vectorized_data, expected)

    print(x_train)
