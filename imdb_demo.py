import os
import argparse
import random
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# set gputf.keras.layers.Layer
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# set random seed
seed = 6222020
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = "1"

from imdb import get_imdb_datasets
from model import InvRNN
from utils import get_pretained_glove
from train import train_imdb
from evaluate import test_imdb

parser = argparse.ArgumentParser(description="invariant rationalization demo.")

# dataset parameters
parser.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help='Path of the dataset')

parser.add_argument('--tr_pollution',
                    nargs="+",
                    type=float,
                    default=[0.7, 0.9],
                    help='Pollution rates of train set [default: [0.7, 0.9]]')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=300,
                    help='Max sequence length [default: 300]')

parser.add_argument('--word_thres',
                    type=int,
                    default=2,
                    help='Min frequency to keep a word [default: 2]')

parser.add_argument('--batch_size',
                    type=int,
                    default=500,
                    help='Batch size [default: 500]')

# pretrained embeddings
parser.add_argument('--glove_path',
                    type=str,
                    default=None,
                    help='Path of pretrained glove embedding [default: None]')

# model parameters
parser.add_argument('--embedding_dim',
                    type=int,
                    default=100,
                    help='Embedding dims [default: 100]')

parser.add_argument('--rnn_dim',
                    type=int,
                    default=100,
                    help='RNN hidden dims [default: 100]')

parser.add_argument('--num_classes',
                    type=int,
                    default=2,
                    help='Number of predicted classes [default: 2]')

parser.add_argument('--num_envs',
                    type=int,
                    default=2,
                    help='Number of environments [default: 2]')

# learning parameters
parser.add_argument('--num_epochs',
                    type=int,
                    default=40,
                    help='Number of training epochs [default: 40]')

parser.add_argument('--sparsity_percentage',
                    type=float,
                    default=0.2,
                    help='Desired highlight percentage [default: .2]')

parser.add_argument('--sparsity_lambda',
                    type=float,
                    default=1.,
                    help='Sparsity trade-off [default: 1.]')

parser.add_argument('--continuity_lambda',
                    type=float,
                    default=5.,
                    help='Continuity trade-off [default: 5.]')

parser.add_argument('--diff_lambda',
                    type=float,
                    default=10.,
                    help='Invariance trade-off [default: 10.]')

parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='Learning rate [default: 1e-3]')

args = parser.parse_args()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# load dataset
######################
args.vocab, D_tr, D_dev, D_te = get_imdb_datasets(args.data_dir,
                                                  args.tr_pollution)

D_tr = D_tr.shuffle(100000, seed=seed,
                    reshuffle_each_iteration=True).batch(args.batch_size,
                                                         drop_remainder=False)
D_dev = D_dev.batch(args.batch_size, drop_remainder=False)
D_te = D_te.batch(args.batch_size, drop_remainder=False)

######################
# Get pretrained embedding
######################
args.pretrained_embedding = get_pretained_glove(args.vocab.word2idx,
                                                args.glove_path)

######################
# build the model
######################
inv_rnn = InvRNN(args)

######################
# optimizer
######################
gen_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
env_inv_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
env_enable_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

opts = [gen_opt, env_inv_opt, env_enable_opt]

global_step = 0

######################
# learning
######################
dev_results = []
te_results = []

for epoch in range(args.num_epochs):
    print("=========================")
    print("epoch:", epoch)
    print("=========================")
    global_step = train_imdb(D_tr, inv_rnn, opts, global_step, args)

    # dev
    inv_acc, _, bias = test_imdb(D_dev, inv_rnn)
    dev_results.append([inv_acc, bias])

    # test
    inv_acc, _, bias = test_imdb(D_te, inv_rnn)
    te_results.append([inv_acc, bias])

np_dev_results = np.array(dev_results)
np_te_results = np.array(te_results)

# check the best dev result
best_dev_epoch = np.argmax(np_dev_results, axis=0)[0]
best_dev_result = np_dev_results[best_dev_epoch, :]

# best corresponding test results
best_cors_te_result = np_te_results[best_dev_epoch, :]

print("=========================")
print("{:s}{:d}.".format("The best dev performance appears at epoch: ",
                         best_dev_epoch),
      flush=True)
print("{:s}{:.4f}, and {:.4f}.".format(
    "The best dev performance and bias selection are: ", best_dev_result[0],
    best_dev_result[1]),
      flush=True)
print("{:s}{:.4f}, and {:.4f}.".format(
    "The corresponding test performance and bias selection are: ",
    best_cors_te_result[0], best_cors_te_result[1]),
      flush=True)
