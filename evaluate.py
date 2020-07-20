import tensorflow as tf


def test_imdb(dataset, model):
    """
    Conventional testing of a classifier.
    """
    avg_env_inv_acc = tf.contrib.eager.metrics.Accuracy("avg_env_inv_acc",
                                                        dtype=tf.float32)
    avg_env_enable_acc = tf.contrib.eager.metrics.Accuracy("avg_env_enable_acc",
                                                           dtype=tf.float32)

    selected_bias = 0
    num_sample = 0

    for (batch, (inputs, masks, labels, envs)) in enumerate(dataset):

        rationale, env_inv_logits, env_enable_logits = model(
            inputs, masks, envs)

        avg_env_inv_acc(tf.argmax(env_inv_logits, axis=1, output_type=tf.int64),
                        tf.argmax(labels, axis=1, output_type=tf.int64))
        avg_env_enable_acc(
            tf.argmax(env_enable_logits, axis=1, output_type=tf.int64),
            tf.argmax(labels, axis=1, output_type=tf.int64))

        # calculate the percentage that the added bias term is highlighted
        selected_bias += tf.reduce_sum(rationale[:, 0, 1])
        num_sample += inputs.get_shape().as_list()[0]

    bias_ = selected_bias / float(num_sample)

    print("{:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Eval] env inv acc: ", avg_env_inv_acc.result(),
        "env enable acc: ", avg_env_enable_acc.result(), "bias selection: ",
        bias_),
          flush=True)

    return avg_env_inv_acc.result(), avg_env_enable_acc.result(), bias_


def test_beer(dataset, model):
    """
    Conventional testing on beer review.
    """
    avg_env_inv_acc = tf.contrib.eager.metrics.Accuracy("avg_env_inv_acc",
                                                        dtype=tf.float32)
    avg_env_enable_acc = tf.contrib.eager.metrics.Accuracy("avg_env_enable_acc",
                                                           dtype=tf.float32)

    for (batch, (inputs, masks, labels, envs)) in enumerate(dataset):

        rationale, env_inv_logits, env_enable_logits = model(
            inputs, masks, envs)

        avg_env_inv_acc(tf.argmax(env_inv_logits, axis=1, output_type=tf.int64),
                        tf.argmax(labels, axis=1, output_type=tf.int64))
        avg_env_enable_acc(
            tf.argmax(env_enable_logits, axis=1, output_type=tf.int64),
            tf.argmax(labels, axis=1, output_type=tf.int64))

    return avg_env_inv_acc.result(), avg_env_enable_acc.result()


def compute_micro_stats(labels, predictions):
    """
    Inputs:
        labels binary sequence indicates the if it is rationale
        predicitions -- sequence indicates the probability of being rationale
    
        labels -- (batch_size, sequence_length) 
        predictions -- (batch_size, sequence_length) in soft probability
    
    Outputs:
        Number of true positive among predicition (True positive)
        Number of predicted positive (True pos + false pos)
        Number of real positive in the labels (true pos + false neg)
    """
    labels = tf.cast(labels, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    # threshold predictions
    predictions = tf.cast(tf.greater_equal(predictions, 0.5), tf.float32)

    # cal precision, recall
    num_true_pos = tf.reduce_sum(labels * predictions)
    num_predicted_pos = tf.reduce_sum(predictions)
    num_real_pos = tf.reduce_sum(labels)

    return num_true_pos, num_predicted_pos, num_real_pos


def validate_beer(dataset, model):
    """
    Compared to annoation.
    """
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    for (batch, (inputs, masks, labels, envs,
                 annotations)) in enumerate(dataset):
        batch_size = inputs.get_shape()[0]

        # go through model
        # rationales -- (batch_size, seq_length, 2)
        rationale, env_inv_logits, env_enable_logits = model(
            inputs, masks, envs)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationale[:, :, 1])

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += tf.reduce_sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / tf.cast(num_words, tf.float32)

    return sparsity, micro_precision, micro_recall, micro_f1
