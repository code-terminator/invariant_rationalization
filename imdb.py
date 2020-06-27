import os
import six
import csv
import math
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

    with open(fpath, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            if n < 0:
                # the header of the CSV files
                n += 1
                continue

            t = convert_to_unicode(line[0])
            y = float(convert_to_unicode(line[1]))

            ts.append(t)
            ys.append(y)

            n += 1

    print("Number of examples %d" % n)

    return np.array(ts), np.array(ys, dtype=np.float32)


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


def pollute_data(t, y, pollution):
    """
    Pollute dataset. 
    Inputs:
        t -- texts (np array)
        y -- labels (np array)
        pollution -- a list of pollution rate for different envs
            if 2 envs total, e.g. [0.3, 0.7]
    """
    num_envs = len(pollution)

    pos_idx = np.where(y > 0.)[0]
    neg_idx = np.where(y == 0.)[0]

    # shaffle these indexs
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # obtain how many pos & neg examples per env
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)

    n = math.floor(num_pos / num_envs)
    num_pos_per_env = np.array(
        [n if i != num_envs - 1 else num_pos - n * i for i in range(num_envs)])
    assert (np.sum(num_pos_per_env) == num_pos)

    n = math.floor(num_neg / num_envs)
    num_neg_per_env = np.array(
        [n if i != num_envs - 1 else num_neg - n * i for i in range(num_envs)])
    assert (np.sum(num_neg_per_env) == num_neg)

    # obtain the pos_idx and neg_idx for each envs
    env_pos_idx = []
    env_neg_idx = []

    s = 0
    for i, num_pos in enumerate(num_pos_per_env):
        idx = pos_idx[s:s + int(num_pos)]
        env_pos_idx.append(set(idx))
        s += int(num_pos)

    s = 0
    for i, num_neg in enumerate(num_neg_per_env):
        idx = neg_idx[s:s + int(num_neg)]
        env_neg_idx.append(set(idx))
        s += int(num_neg)

    # create a lookup table idx --> env_id
    idx2env = {}

    for env_id, idxs in enumerate(env_pos_idx):
        for idx in idxs:
            idx2env[idx] = env_id
    assert (len(idx2env.keys()) == len(pos_idx))

    for env_id, idxs in enumerate(env_neg_idx):
        for idx in idxs:
            idx2env[idx] = env_id
    assert (len(idx2env.keys()) == len(t))

    new_t = []
    envs = []

    for idx, t_ in enumerate(t):
        env_id = idx2env[idx]
        rate = pollution[env_id]

        envs.append(env_id)

        if np.random.choice([0, 1], p=[1. - rate, rate]) == 1:
            if y[idx] == 1.:
                text = ", " + t_
            else:
                text = ". " + t_
        else:
            if y[idx] == 1.:
                text = ". " + t_
            else:
                text = ", " + t_

        new_t.append(text)

    return new_t, envs


def get_imdb_datasets(data_dir, tr_pollution, max_seq_length=300, word_thres=2):
    """
    Generate polluted imdb datasets.
    """
    ### training set
    print("Training set: ")
    t_tr, y_tr = get_examples(os.path.join(data_dir, "train.tsv"))

    ### dev set
    print("Dev set: ")
    t_dev_, y_dev_ = get_examples(os.path.join(data_dir, "dev.tsv"))

    ##### further split the provided dev into dev and test set
    pos_idx = np.where(y_dev_ > 0.)[0]

    neg_idx = np.where(y_dev_ == 0.)[0]

    dev_idx = np.concatenate(
        [pos_idx[0:len(pos_idx) // 2], neg_idx[0:len(neg_idx) // 2]],
        axis=0).astype(np.int32)
    te_idx = np.concatenate(
        [pos_idx[len(pos_idx) // 2::], neg_idx[len(neg_idx) // 2::]],
        axis=0).astype(np.int32)

    assert (set(dev_idx.tolist() + te_idx.tolist()) == set(range(len(y_dev_))))

    t_dev = t_dev_[dev_idx]
    y_dev = y_dev_[dev_idx]

    t_te = t_dev_[te_idx]
    y_te = y_dev_[te_idx]

    ##### pollute train, dev, and test sets accordingly and creates different envs
    pt_tr, e_tr = pollute_data(t_tr, y_tr, tr_pollution)
    pt_dev, e_dev = pollute_data(t_dev, y_dev, [0.5, 0.5])
    pt_te, e_te = pollute_data(t_te, y_te, [1. - p for p in tr_pollution])

    ##### constrcut word dictionary
    # this need to be done at the end, because, pollution might add new tokens
    texts = pt_tr + pt_dev + pt_te
    vocab = LanguageIndex(texts, word_thres)

    ##### convert texts to index with proper length

    ### train
    x_tr = []
    m_tr = []

    for example in pt_tr:
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

    for example in pt_dev:
        x, m = text2idx(example, max_seq_length, vocab.word2idx)
        x_dev.append(x)
        m_dev.append(m)

    x_dev = np.array(x_dev, dtype=np.int32)
    m_dev = np.array(m_dev, dtype=np.float32)
    y_dev = binary_one_hot_convert(y_dev).astype(np.float32)
    e_dev = binary_one_hot_convert(e_dev).astype(np.float32)

    ### test
    x_te = []
    m_te = []

    for example in pt_te:
        x, m = text2idx(example, max_seq_length, vocab.word2idx)
        x_te.append(x)
        m_te.append(m)

    x_te = np.array(x_te, dtype=np.int32)
    m_te = np.array(m_te, dtype=np.float32)
    y_te = binary_one_hot_convert(y_te).astype(np.float32)
    e_te = binary_one_hot_convert(e_te).astype(np.float32)

    ##### construct dataset

    D_tr = tf.data.Dataset.from_tensor_slices((x_tr, m_tr, y_tr, e_tr))
    D_dev = tf.data.Dataset.from_tensor_slices((x_dev, m_dev, y_dev, e_dev))
    D_te = tf.data.Dataset.from_tensor_slices((x_te, m_te, y_te, e_te))

    return vocab, D_tr, D_dev, D_te
