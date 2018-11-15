#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : vocabuary_tools.py
# PythonVersion: python3.5
# Date    : 2018/11/14 19:47
# Software: PyCharm

"""Create tools for generate vocabulary in NLP.This was collected from tensorflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

import numpy as np
try:
  # pylint: disable=g-import-not-at-top
  import cPickle as pickle
except ImportError:
  # pylint: disable=g-import-not-at-top
  import pickle


class CategoricalVocabulary(object):
    """Map categories to indexes, can be used for categorical variables, sparse variables and words.
    """
    def __init__(self, unknown_token="<UNK>", support_reverse=True):
        self._unknown_token = unknown_token
        self._support_reverse = support_reverse
        self._mapping = {unknown_token: 0}
        if support_reverse:
            self._reverse_mapping = [unknown_token]
        self._freq = collections.defaultdict(int)
        self._freeze = False  # flag to control whether to update dict

    def __len__(self):
        """Returns total count of mappings, including unknown token."""
        return len(self._mapping)

    def freeze(self, freeze=True):
        """
        Freezes the vocabulary, if freeze the new word will return unknown id.
        :param freeze: True to lock vocabulary, False not
        :return:
        """
        self._freeze = freeze

    def get(self, category):
        """Return word's id in the vocabulary, if category was new create new id for it.
        :param category, the word or category want to found in the vocabulary
        :return word's id in the vocabulary
        """
        if category not in self._mapping:
            if self._freeze:  # Freeze dict, not insert new word return unknown id
                return 0
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)
        return self._mapping[category]

    def add(self, category, count=1):
        """
        Add count of the category to the frequency dict.
        :param category: input word
        :param count: term frequency
        :return:
        """
        category_id = self.get(category)
        if category_id <= 0:   # if category not in dictionary, frequency add nothing
            return
        self._freq[category_id] += count

    def trim(self, min_frequency, max_frequency=-1):
        """
        Remove words form the vocabulary based on the minimum frequency.
        :param min_frequency: minimum frequency to keep
        :param max_frequency: maximum frequency to keep (optional), usefull for the high frequency terms
        :return:
        """
        # sort word frequency dictionary
        # here were two way to complete sort, firstly sort the by key ,second sort by frequency
        # Version 1
        self._freq = sorted(sorted(six.iteritems(self._freq), key= lambda x: (isinstance(x[0], str), x[0])),
                            key= lambda x: x[1], reverse=True)
        # Version 2
        # self._freq = sorted(six.iteritems(self._freq), key= lambda x: x[1], reverse=True)

        # remove and re-assign index
        self._mapping = {self._unknown_token: 0}
        if self._support_reverse:
            self._reverse_mapping = [self._unknown_token]
        idx = 1
        for category, count in self._freq:
            # check count bigger than maximum frequency
            if 0 < max_frequency <= count:
                continue
            if count <= min_frequency:
                break
            self._mapping[category] = idx
            idx += 1
            if self._support_reverse:
                self._reverse_mapping.append(category)
        # word frequency dict was reverse sorted so only choose the (0, idx -1) elements
        self._freq = self._freq[: idx -1]

    def reverse(self, class_id):
        """
        Given a word id to find it's original word.
        :param class_id: word id
        :return: original word
        """
        if not self._support_reverse:
            raise ValueError("This vocabulay wasn't initialized with support reverse.")
        return self._reverse_mapping[class_id]


# create vocabulary processor
class VocabularyProcessor(object):
    """Vocabulary processor help create vocabulary and map sequence to word ids.
    """
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None,
                 tokenizer_fn=None):
        """
        Initialize vocabulary processor
        :param max_document_length: maximum length of document for sequence padding, if document length are longer they
                will be trimmed, if shorter will be padded
        :param min_frequency: minimum term frequency when create vocabulary
        :param vocabulary: category vocabulary object (class defined upon
        :param tokenizer_fn: tokenizer function
        """
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        # vocabulary was create new object
        if vocabulary:
            self.vocabulary = vocabulary
        else:
            self.vocabulary = CategoricalVocabulary()

        if tokenizer_fn:
            self._tokenizer = tokenizer_fn
        else:
            print("You can create your own tokenizer.")

    def fit(self, raw_documents):
        """
        Learn a vocabulary dictionary of all tokens in raw documents
        :param raw_documents: An iterable which yield either string or unicode.
        :return: self
        """
        # Extract tokens and add to vocabulary
        for tokens in self._tokenizer(raw_documents):
            for token in tokens:
                self.vocabulary.add(token)

        # clean
        if self.min_frequency > 0:
            self.vocabulary.trim(self.min_frequency)
        # lock vocabulary
        self.vocabulary.freeze()
        return self

    def transform(self, raw_documents):
        """
        Transform documents to word-id matrix.Convert words to ids with vocabulary fitted with fit or the one
        provided in constructor
        :param raw_documents: An iterable which
        :return: Yeilds: x: iterable, shape = [n_samples, max_document_length], word-id matrix
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx > self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary.get(token)
            yield word_ids

    def fit_transform(self, raw_documents):
        """
        Learn the vocabulary of all tokens in raw documents and convert document to ids
        :param raw_documents: An iterable which yield either string or unicode
        :return: convert tokens to ids in documents, shape = [n_samples, max_document_length], word-id matrix
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def reverse(self, documents):
        """
        Reverse output of vocabulary mapping to words
        :param documents: iterable , list of documents
        :return: Yields, Iterator over mapped in words documents
        """
        for item in documents:
            output = []
            for class_id in item:
                output.append(self.vocabulary.reverse(class_id))
            yield ' '.join(output)

    def save(self,filename):
        """
        Save vocabulary processor into given file
        :param filename: saved filename
        :return:
        """
        with open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @classmethod
    def restore(cls, filename):
        """
        Restore vocabulary processor from given file
        :param filename: file name
        :return: Vocabulary processor object
        """
        with open(filename, 'rb') as f:
            return pickle.loads(f.read())