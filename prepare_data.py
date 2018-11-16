#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : prepare_data.py
# PythonVersion: python3.5
# Date    : 2018/11/14 17:25
# Software: PyCharm

"""An useful tools to prepare data for the model train/develop and test. It will generate three TFRecord files,
meanwhile create vocabulary.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

import os
import functools
import itertools
import numpy as np
import csv

import jieba
from vocabulary_tools import VocabularyProcessor, is_need_remove

# define input arguments
tf.flags.DEFINE_integer("min_word_frequency", 5, "the minmum word frequency in dataset")
# TODO, use bucket mechanism
tf.flags.DEFINE_integer("max_document_length", 200, "the maximum sentence length.")
tf.flags.DEFINE_string("data_dir", "/data/research/data/chat/DoubanConversaionCorpus",
                       "data directory of train/dev/test")
tf.flags.DEFINE_string("output_dir", "." ,"directory of preocessed data")

FLAGS = tf.flags.FLAGS

train_dir = FLAGS.data_dir + "/new_train.txt"
valid_dir = FLAGS.data_dir + "/new_dev.txt"
test_dir = FLAGS.data_dir + "/new_test.txt"


# tokenizer
def tokenizer_fn(iterator):
    """
    Tokenizer function
    :param iterator: An iterator of documents
    :return: tokens list
    """
    # return (list(jieba.cut(x)) for x in iterator)
    return (list(x.split(' ')) for x in iterator)


# create iterator
def create_iter(filename):
    """
    Returns an iterator over a file, if csv skip header
    :param filename:
    :return:
    """
    if filename.endswith('.csv'):
        reader = csv.reader(filename)
        next(reader)
        for row in reader:
            yield row
    else:
        reader = open(filename, 'r')
        for row in reader:
            yield row


# create vocab
def create_vocab(input_iter, min_frequency):
    """
    Create vocabulary based on input iterator documents
    :param input_iter: input row iterator
    :param min_frequency: the minimum frequency in corpus
    :return: vocabulary processor object
    """
    vocab_processor = VocabularyProcessor(
        FLAGS.max_document_length,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)
    # create vocabulary
    vocab_processor.fit(input_iter)
    return vocab_processor


# transform sentence
def transform_sentence(sentence, vocab_processor):
    """
    Convert sentence to id sequence
    :param sentence: input sentence
    :param vocab_processor: Vocabulary Processor object
    :return: a id array
    """
    return next(vocab_processor.transform([sentence])).tolist()


# add sentence id list to Feature list
def create_text_sequnce_feature(featureList, sentence, sentence_len, vocab):
    """
    Add sentence to FeatureList protocol Buffer object
    :param featureList: Feature list
    :param sentence: input sentence
    :param sentence_len: sentence length
    :param vocab: vocabulary processor object
    :return:
    """
    transformed = transform_sentence(sentence, vocab)
    for word_id in transformed:
        featureList.feature.add().int64_list.value.extand([word_id])
    return featureList


# create example
def create_example(row, vocab):
    """
    Create example for input data set
    :param row: each line in input file
    :param vocab: vocabulary processor object
    :return: the tensorflow.Example Protocol Buffer object
    """

    # parse row, may modified according to the dataset structure
    # print(row)
    label, context, response = row.split('\t')

    # convert feature list
    context_transformed = transform_sentence(context, vocab)
    response_transformed = transform_sentence(response, vocab)
    context_len = len(next(vocab._tokenizer(context)))
    response_len = len(next(vocab._tokenizer(response)))
    label = int(label)

    # New example
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["response"].int64_list.value.extend(response_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["response_len"].int64_list.value.extend([response_len])
    example.features.feature["label"].int64_list.value.extend([label])

    return example


# save example to TFRecords
def create_tfrecords_file(input_filename, outfile_name, example_fn):
    """
    Create TFRecords for given input file
    :param input_filename: input filename
    :param outfile_name: output TFRecords filename
    :param example_fn: create example function
    :return:
    """
    writer = tf.python_io.TFRecordWriter(outfile_name)
    print("Creating TFRecords file at {}...".format(outfile_name))
    for i, row in enumerate(create_iter(input_filename)):
        row_example = example_fn(row)
        writer.write(row_example.SerializeToString())
    writer.close()
    print("Created TFRecord {} done!".format(outfile_name))


# store vocabulary
def store_vocabulary(vocab_processor, outfile):
    """
    Store vocabulary to outfile
    :param vocab_processor: vocabulary processor object
    :param outfile: output filename
    :return:
    """
    vocab_size = len(vocab_processor.vocabulary)
    with open(outfile, 'w') as f:
        for id in range(vocab_size):
            word = vocab_processor.vocabulary._reverse_mapping[id]
            # print(word, str(id))
            if is_need_remove(word):
                continue
            f.write(word + "\t" + str(id))
            f.write('\n')
    print("Saved vocabulary in {}".format(outfile))


if __name__ == "__main__":
    print("Create vocabulary...")
    # create iterator
    input_iter = create_iter(train_dir)

    # create vocab
    # for x in input_iter:
    #     print(x)
    input_iter = (' '.join(x.split('\t')[1:]) for x in input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary)))

    # store vocabulary into files
    store_vocabulary(vocab, FLAGS.output_dir + '/vocabulary.txt')

    # create train TFRecords
    create_tfrecords_file(
        input_filename=train_dir,
        outfile_name=FLAGS.output_dir + '/tfrecord.train',
        example_fn=functools.partial(create_example, vocab=vocab)
    )

    # create dev TFRecords
    create_tfrecords_file(
        input_filename=valid_dir,
        outfile_name=FLAGS.output_dir + '/tfrecord.dev',
        example_fn=functools.partial(create_example, vocab=vocab)
    )

    # create dev TFRecords
    create_tfrecords_file(
        input_filename=test_dir,
        outfile_name=FLAGS.output_dir + '/tfrecord.test',
        example_fn=functools.partial(create_example, vocab=vocab)
    )

