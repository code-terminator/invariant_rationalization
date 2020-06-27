import sys
import sty
import numpy as np
import tensorflow as tf


def get_pretained_glove(word2idx, fpath):
    """
    Construct a numpy embedding matrix. 
    The column number indicates the word index.
    For the words do not appear in pretrained embeddings, 
    we use random embeddings.
    Inputs:
        word2idx -- a dictionary, key -- word, value -- word index
        fpath -- the path of pretrained embedding.
    Outputs:
        embedding_matrix -- an ordered numpy array, 
                            shape -- (embedding_dim, len(word2idx))
    """

    def load_glove_embedding():
        """
        Load glove embedding from disk. 
        """
        word2embedding = {}
        with open(fpath, "r", errors='ignore') as f:
            for (i, line) in enumerate(f):
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = list(map(float, data[1:]))
                word2embedding[word] = np.array(
                    embedding)  # shape -- (embedding_dim, )
            embedding_dim = len(embedding)

        return word2embedding, embedding_dim

    # load glove embedding
    word2embedding, embedding_dim = load_glove_embedding()
    embedding_matrix = np.random.randn(embedding_dim, len(word2idx))

    # replace the embedding matrix by pretrained embedding
    counter = 0
    for word, index in word2idx.items():
        if word in word2embedding:
            embedding_matrix[:, index] = word2embedding[word]
            counter += 1

    # replace the embedding to all zeros for <pad>
    embedding_matrix[:, word2idx["<pad>"]] = np.zeros(embedding_dim)
    print("%d out of %d words are covered by the pre-trained embedding." %
          (counter, len(word2idx)))

    return embedding_matrix


def inv_rat_loss(env_inv_logits, env_enable_logits, labels):
    """
    Compute the loss for the invariant rationalization training.
    Inputs:
        env_inv_logits -- logits of the predictor without env index
                          (batch_size, num_classes)
        env_enable_logits -- logits of the predictor with env index
                          (batch_size, num_classes)        
        labels -- the groundtruth one-hot labels 
    """
    env_inv_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=env_inv_logits, labels=labels)

    env_enable_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=env_enable_logits, labels=labels)

    env_inv_loss = tf.reduce_mean(env_inv_losses)
    env_enable_loss = tf.reduce_mean(env_enable_losses)

    diff_loss = tf.math.maximum(0., env_inv_loss - env_enable_loss)

    return env_inv_loss, env_enable_loss, diff_loss


def cal_sparsity_loss(z, mask, level):
    """
    Exact sparsity loss in a batchwise sense. 
    Inputs: 
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """
    sparsity = tf.reduce_sum(z) / tf.reduce_sum(mask)
    return tf.abs(sparsity - level)


def cal_continuity_loss(z):
    """
    Compute the continuity loss.
    Inputs:     
        z -- (batch_size, sequence_length)
    """
    return tf.reduce_mean(tf.abs(z[:, 1:] - z[:, :-1]))


def show_binary_rationale(ids, z, idx2word, tofile=False):
    """
    Visualize rationale.  
    Inputs:
        ids -- numpy of the text ids (sequence_length,).
        z -- binary rationale (sequence_length,).
        idx2word -- map id to word.
    """
    text = [idx2word[idx] for idx in ids]
    output = ""
    for i, word in enumerate(text):
        if z[i] == 1:
            output += sty.fg.red + word + sty.fg.rs + " "
        else:
            output += word + " "

    if tofile:
        return output
    else:
        try:
            print(output)
        except Exception as e:
            print(e)

        sys.stdout.flush()
