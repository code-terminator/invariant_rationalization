import os
import six
import csv
import math
import json
import numpy as np
import tensorflow as tf


class LanguageIndex(object):
    """
    Creates a word -> index mapping (e.g,. "dad" -> 5) 
    and vice-versa.
    """

    def __init__(self, texts, threshold=1):
        """
        Inputs: 
            texts -- a list of text (after tokenization)
            threshold -- threshold to filter less frequent words
        """
        self.threshold = threshold

        self.word2idx = {}
        self.idx2word = {}
        self._create_index(texts)

    def _create_index(self, texts):

        # counting for unique words
        word2count = {}
        for text in texts:
            for word in text.split(' '):
                if word in word2count:
                    word2count[word] += 1
                else:
                    word2count[word] = 1

        # counting unqiue words
        vocab = set()
        for word, count in word2count.items():
            if count >= self.threshold:
                vocab.add(word)
        vocab = sorted(vocab)

        # create word2idx
        self.word2idx["<pad>"] = 0
        self.word2idx["<unknown>"] = 1
        for index, word in enumerate(vocab):
            self.word2idx[word] = index + 2

        # create reverse index
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def convert_to_unicode(text):
    """
    Converts text to Unicode (if it's not already)
    assuming utf-8 input.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def get_examples(fpath):
    """
    Get data from a tsv file.
    Input:
        fpath -- the file path.
    """
    n = -1
    ts = []
    ys = []
    es = []

    with open(fpath, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            if n < 0:
                # the header of the CSV files
                n += 1
                continue

            t = convert_to_unicode(line[0])
            y = float(convert_to_unicode(line[1]))
            e = float(convert_to_unicode(line[2]))

            ts.append(t)
            ys.append(y)
            es.append(e)

            n += 1

    print("Number of examples %d" % n)

    return ts, np.array(ys, dtype=np.float32), np.array(es, dtype=np.float32)


def text2idx(text, max_seq_length, word2idx):
    """
    Converts a single text into a list of ids with mask. 
    This function consider annotaiton of z1, z2, z3 are provided
    """
    input_ids = []

    text_ = text.strip().split(" ")

    if len(text_) > max_seq_length:
        text_ = text_[0:max_seq_length]

    for word in text_:
        word = word.strip()
        try:
            input_ids.append(word2idx[word])
        except:
            # if the word is not exist in word2idx, use <unknown> token
            input_ids.append(1)

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    # zero-pad up to the max_seq_length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask


def binary_one_hot_convert(list_of_label):
    """
    Convert a list of binary label to one_hot numpy format.
    """
    np_label = np.expand_dims(np.array(list_of_label, dtype=np.float32), axis=1)
    one_hot_label = np.concatenate([1. - np_label, np_label], axis=1)
    return one_hot_label


def get_beer_datasets(data_dir, max_seq_length=300, word_thres=2):
    """
    Generate beer review datasets.
    """
    ### training set
    print("Training set: ")
    t_tr, y_tr, e_tr = get_examples(os.path.join(data_dir, "train.tsv"))

    ### dev set
    print("Dev set: ")
    t_dev, y_dev, e_dev = get_examples(os.path.join(data_dir, "dev.tsv"))

    ##### constrcut word dictionary
    texts = t_tr + t_dev
    vocab = LanguageIndex(texts, word_thres)

    ##### convert texts to index with proper length

    ### train
    x_tr = []
    m_tr = []

    for example in t_tr:
        x, m = text2idx(example, max_seq_length, vocab.word2idx)
        x_tr.append(x)
        m_tr.append(m)

    x_tr = np.array(x_tr, dtype=np.int32)
    m_tr = np.array(m_tr, dtype=np.float32)
    y_tr = binary_one_hot_convert(y_tr).astype(np.float32)
    e_tr = binary_one_hot_convert(e_tr).astype(np.float32)

    ### dev
    x_dev = []
    m_dev = []

    for example in t_dev:
        x, m = text2idx(example, max_seq_length, vocab.word2idx)
        x_dev.append(x)
        m_dev.append(m)

    x_dev = np.array(x_dev, dtype=np.int32)
    m_dev = np.array(m_dev, dtype=np.float32)
    y_dev = binary_one_hot_convert(y_dev).astype(np.float32)
    e_dev = binary_one_hot_convert(e_dev).astype(np.float32)

    ##### construct dataset
    D_tr = tf.data.Dataset.from_tensor_slices((x_tr, m_tr, y_tr, e_tr))
    D_dev = tf.data.Dataset.from_tensor_slices((x_dev, m_dev, y_dev, e_dev))

    return vocab, D_tr, D_dev


def get_beer_annotation(data_dir, aspect, max_seq_length, word2idx):
    """
    Read annotation from json and 
    return tf datasets of the beer annotation.
    """
    annotation_path = os.path.join(data_dir, "annotations.json")

    data = []
    labels = []
    envs = []
    masks = []
    rationales = []
    num_classes = 2

    with open(annotation_path, "rt") as fin:
        for counter, line in enumerate(fin):
            item = json.loads(line)

            # obtain the data
            text_ = item["x"]
            ratings = item["y"][:-1]
            y = ratings[aspect]
            env = ratings[:aspect] + ratings[aspect + 1:]
            rationale = item[str(aspect)]

            # check if the rationale is all zero
            if len(rationale) == 0:
                # no rationale for this aspect
                continue

            # process the label
            if float(y) >= 0.6:
                y = 1
            elif float(y) <= 0.4:
                y = 0
            else:
                continue
            one_hot_label = [0] * num_classes
            one_hot_label[y] = 1

            # process the text
            input_ids = []
            if len(text_) > max_seq_length:
                text_ = text_[0:max_seq_length]

            for word in text_:
                word = word.strip()
                try:
                    input_ids.append(word2idx[word])
                except:
                    # word is not exist in word2idx, use <unknown> token
                    input_ids.append(word2idx["<unknown>"])

            # process mask
            # The mask has 1 for real word and 0 for padding tokens.
            input_mask = [1] * len(input_ids)

            # zero-pad up to the max_seq_length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)

            assert (len(input_ids) == max_seq_length)
            assert (len(input_mask) == max_seq_length)

            # construct rationale
            binary_rationale = [0] * len(input_ids)
            for zs in rationale:
                start = zs[0]
                end = zs[1]
                if start >= max_seq_length:
                    continue
                if end >= max_seq_length:
                    end = max_seq_length

                for idx in range(start, end):
                    binary_rationale[idx] = 1

            data.append(input_ids)
            labels.append(one_hot_label)
            envs.append(env)
            masks.append(input_mask)
            rationales.append(binary_rationale)

        data = np.array(data, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        envs = np.array(envs, dtype=np.float32)
        masks = np.array(masks, dtype=np.int32)
        rationales = np.array(rationales, dtype=np.int32)

        print("Annotated rationales: %d" % data.shape[0])

        annotated_dataset = tf.data.Dataset.from_tensor_slices(
            (data, masks, labels, envs, rationales))

    return annotated_dataset
