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
