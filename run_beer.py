import os
import argparse
import random
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from beer import get_beer_datasets, get_beer_annotation
from utils import get_pretained_glove
from beer_model import InvRNNwithSpanPred
from train import train_beer
from evaluate import test_beer, validate_beer

parser = argparse.ArgumentParser(
    description="run invariant rationalization on beer review.")

# dataset parameters
parser.add_argument('--aspect',
                    type=int,
                    required=True,
                    help='Aspect of the beer')

parser.add_argument('--base_dir',
                    type=str,
                    required=True,
                    help='Base dir of the dataset')

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
                    default=256,
                    help='RNN hidden dims [default: 256]')

parser.add_argument('--num_classes',
                    type=int,
                    default=2,
                    help='Number of predicted classes [default: 2]')

parser.add_argument('--num_envs',
                    type=int,
                    default=2,
                    help='Number of environments [default: 2]')

parser.add_argument('--rationale_length',
                    type=int,
                    default=10,
                    help='Number of environments [default: 10]')

# learning parameters
parser.add_argument('--num_epochs',
                    type=int,
                    default=50,
                    help='Number of training epochs [default: 50]')

parser.add_argument('--diff_lambda',
                    type=float,
                    default=1.,
                    help='Invariance trade-off [default: 1.]')

parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='Learning rate [default: 1e-3]')

# output file
parser.add_argument('--output_file',
                    type=str,
                    required=True,
                    help='Output file name')

# gpu
parser.add_argument('--gpu',
                    type=str,
                    default="0",
                    help='id(s) for CUDA_VISIBLE_DEVICES [default: 0]')

args = parser.parse_args()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# set visiable gpu
######################
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

######################
# load dataset
######################
data_dir = os.path.join(args.base_dir, "aspect_%d" % args.aspect)

args.vocab, D_tr_, D_dev = get_beer_datasets(data_dir, args.max_seq_length,
                                             args.word_thres)
D_ann = get_beer_annotation(args.base_dir, args.aspect, args.max_seq_length,
                            args.vocab.word2idx)

D_tr = D_tr_.shuffle(100000).batch(args.batch_size, drop_remainder=False)
D_dev = D_dev.batch(args.batch_size, drop_remainder=False)
D_ann = D_ann.batch(args.batch_size, drop_remainder=False)

######################
# Get pretrained embedding
######################
args.pretrained_embedding = get_pretained_glove(args.vocab.word2idx,
                                                args.glove_path)

######################
# build the model
######################
inv_rnn = InvRNNwithSpanPred(args)

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
ann_results = []

for epoch in range(args.num_epochs):
    print("=========================")
    print("epoch:", epoch)
    # reshuffle the dataset
    D_tr = D_tr_.shuffle(100000).batch(args.batch_size, drop_remainder=False)

    global_step = train_beer(D_tr, inv_rnn, opts, global_step, args)

    # dev
    dev_result = test_beer(D_dev, inv_rnn)
    dev_results.append(list(dev_result))

    # ann
    ann_result = validate_beer(D_ann, inv_rnn)
    ann_results.append(list(ann_result))

np_dev_results = np.array(dev_results)
np_ann_results = np.array(ann_results)

# check the best dev result
best_dev_epoch = np.argmax(np_dev_results, axis=0)[0]
best_dev_result = np_dev_results[best_dev_epoch, :]
best_cors_ann_result = np_ann_results[best_dev_epoch, :]

print(
    "{:s}{:d}, {:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}."
    .format("----> [Overall] Best dev occurs at epoch: ", best_dev_epoch,
            "dev inv acc: ", best_dev_result[0], "dev enb acc: ",
            best_dev_result[1], "The corresponding annotation sparsity: ",
            best_cors_ann_result[0], "precision: ", best_cors_ann_result[1],
            "recall: ", best_cors_ann_result[2], "f1: ",
            best_cors_ann_result[3]),
    flush=True)

output_string = "%d\t%f\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
    args.rationale_length, args.diff_lambda, best_dev_epoch,
    best_dev_result[0] * 100, best_dev_result[1] * 100,
    100 * best_cors_ann_result[0], 100 * best_cors_ann_result[1],
    100 * best_cors_ann_result[2], 100 * best_cors_ann_result[3])

f = open(args.output_file, "a")
f.write(output_string)
f.close()
