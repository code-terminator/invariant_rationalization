import tensorflow as tf
from utils import inv_rat_loss, cal_sparsity_loss, cal_continuity_loss, show_binary_rationale


def train_imdb(dataset, model, opts, step, args, show_rationale=False):
    """
    Training invariant model on imdb.
    """
    ### obtain the optimizer
    gen_opt = opts[0]
    env_inv_opt = opts[1]
    env_enable_opt = opts[2]

    ### average loss
    avg_env_inv_acc = tf.contrib.eager.metrics.Accuracy("avg_env_inv_acc",
                                                        dtype=tf.float32)
    avg_env_enable_acc = tf.contrib.eager.metrics.Accuracy("avg_env_enable_acc",
                                                           dtype=tf.float32)

    selected_bias = 0
    num_sample = 0

    for (batch, (inputs, masks, labels, envs)) in enumerate(dataset):

        path = step % 7

        with tf.GradientTape() as tape:

            rationale, env_inv_logits, env_enable_logits = model(
                inputs, masks, envs)

            env_inv_loss, env_enable_loss, diff_loss = inv_rat_loss(
                env_inv_logits, env_enable_logits, labels)

            sparsity_loss = args.sparsity_lambda * cal_sparsity_loss(
                rationale[:, :, 1], masks, args.sparsity_percentage)

            continuity_loss = args.continuity_lambda * cal_continuity_loss(
                rationale[:, :, 1])

            gen_loss = args.diff_lambda * diff_loss + env_inv_loss
            gen_loss += sparsity_loss + continuity_loss

        # corrdinate descent
        # apply gradient based on `path`
        if path == 0:
            # update the generator
            gen_vars = model.generator_trainable_variables()
            gen_grads = tape.gradient(gen_loss, gen_vars)
            gen_opt.apply_gradients(zip(gen_grads, gen_vars))
        elif path in [1, 2, 3]:
            # update the env inv predictor
            env_inv_vars = model.env_inv_trainable_variables()
            env_inv_grads = tape.gradient(env_inv_loss, env_inv_vars)
            env_inv_opt.apply_gradients(zip(env_inv_grads, env_inv_vars))
        elif path in [4, 5, 6]:
            # update the env enabled predictor
            env_enable_vars = model.env_enable_trainable_variables()
            env_enable_grads = tape.gradient(env_enable_loss, env_enable_vars)
            env_enable_opt.apply_gradients(
                zip(env_enable_grads, env_enable_vars))
        else:
            raise ValueError("Invalid path number. ")

        step += 1

        avg_env_inv_acc(tf.argmax(env_inv_logits, axis=1, output_type=tf.int64),
                        tf.argmax(labels, axis=1, output_type=tf.int64))
        avg_env_enable_acc(
            tf.argmax(env_enable_logits, axis=1, output_type=tf.int64),
            tf.argmax(labels, axis=1, output_type=tf.int64))

        # calculate the percentage that the added bias term is highlighted
        selected_bias += tf.reduce_sum(rationale[:, 0, 1])
        num_sample += inputs.get_shape().as_list()[0]

    bias_ = selected_bias / float(num_sample)

    # visualize
    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Train] Total iteration #", step, "env inv acc: ",
        avg_env_inv_acc.result(), "env enable acc: ",
        avg_env_enable_acc.result(), "bias selection: ", bias_),
          flush=True)

    # show a generated rationale
    if show_rationale:
        print("----> [Train] visualizing rationale: groundtruth label: {:d}.".
              format(tf.argmax(labels[0, :])),
              flush=True)
        show_binary_rationale(inputs[0, :].numpy(), rationale[0, :, 1].numpy(),
                              args.vocab.idx2word)

    return step
