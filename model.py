import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    """
    The embedding layer. 
    """

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        """
        Inputs:
            pretrained_embedding -- a numpy array (embedding_dim, vocab_size)
        """
        super(Embedding, self).__init__()

        try:
            init = tf.keras.initializers.Constant(pretrained_embedding)
            print("Initialize the embedding from a pre-trained matrix.")
        except:
            init = "uniform"
            print("Initialize the embedding randomly.")

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   embeddings_initializer=init)

    def call(self, x):
        """
        Inputs:
            x -- (batch_size, seq_length)
        Outputs:
            shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


class RnnEncoder(tf.keras.layers.Layer):
    """
    The RNN encoder for document disentanglement.
    """

    def __init__(self, hdim):
        super(RnnEncoder, self).__init__()

        self.hdim = hdim

        self.rnn = tf.compat.v1.keras.layers.CuDNNGRU(units=hdim,
                                                      return_sequences=True)

        self.rnn = tf.keras.layers.Bidirectional(self.rnn)

    def call(self, x):
        """
        Inputs: 
            x -- (batch_size, seq_length, input_dim)
        Outputs: 
            y -- bidirectional (batch_size, seq_length, hidden_dim * 2)
        """
        h = self.rnn(x)

        return h


class InvRNN(tf.keras.Model):
    """
    A RNN-based invariant rationalization model.
    """

    def __init__(self, args):
        super(InvRNN, self).__init__()

        self.args = args

        # initialize three embedding layers
        self.gen_embed_layer = Embedding(len(args.vocab.word2idx),
                                         args.embedding_dim,
                                         args.pretrained_embedding)

        self.env_inv_embed_layer = Embedding(len(args.vocab.word2idx),
                                             args.embedding_dim,
                                             args.pretrained_embedding)

        self.env_enable_embed_layer = Embedding(len(args.vocab.word2idx),
                                                args.embedding_dim,
                                                args.pretrained_embedding)

        # initialize the rationale generator
        self.generator = RnnEncoder(hdim=args.rnn_dim)

        # initialize the RNN encoders for the predictors
        self.env_inv_encoder = RnnEncoder(hdim=args.rnn_dim)

        self.env_enable_encoder = RnnEncoder(hdim=args.rnn_dim)

        # generator output layer (binary selection)
        self.generator_fc = tf.keras.layers.Dense(units=2)

        # encoder output layer (classification task)
        self.env_inv_fc = tf.keras.layers.Dense(units=args.num_classes)
        self.env_enable_fc = tf.keras.layers.Dense(units=args.num_classes)

    def _independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)        
        """
        z = tf.nn.softmax(rationale_logits)
        z_hard = tf.cast(tf.equal(z, tf.reduce_max(z, -1, keep_dims=True)),
                         z.dtype)
        z = tf.stop_gradient(z_hard - z) + z

        return z

    def call(self, inputs, masks, envs):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
            envs -- (batch_size, num_envs)
        """
        # expand dim for masks
        masks_ = tf.cast(tf.expand_dims(masks, -1), tf.float32)

        all_ones = tf.expand_dims(tf.ones(inputs.shape), axis=-1)
        all_zeros = tf.zeros(all_ones.shape)

        ############## generator ##############
        gen_embeddings = masks_ * self.gen_embed_layer(inputs)
        gen_outputs = self.generator(gen_embeddings)
        gen_logits = self.generator_fc(gen_outputs)

        # sample rationale (batch_size, sequence_length, 2)
        rationale = self._independent_straight_through_sampling(gen_logits)

        # mask the rationale that corresponding to <pad>
        rationale = masks_ * rationale + (1. - masks_) * tf.concat(
            [all_ones, all_zeros], axis=-1)

        ############## env inv predictor ##############
        env_inv_embeddings = masks_ * self.env_inv_embed_layer(inputs)
        env_inv_rat_embeddings = env_inv_embeddings * tf.expand_dims(
            rationale[:, :, 1], 2)

        env_inv_enc_outputs = self.env_inv_encoder(env_inv_rat_embeddings)

        # mask before max pooling
        # (1 - mask) * (-1e9) --> <pad> become very neg
        env_inv_enc_outputs_ = env_inv_enc_outputs * masks_ + (1. -
                                                               masks_) * (-1e9)

        # aggregates hidden outputs via max pooling
        # shape -- (batch_size, hidden_dim * 2)
        env_inv_enc_output = tf.reduce_max(env_inv_enc_outputs_, axis=1)

        # task prediction
        # shape -- (batch_size, num_classes)
        env_inv_logits = self.env_inv_fc(env_inv_enc_output)

        ############## env enable predictor ##############
        env_enable_embeddings = masks_ * self.env_enable_embed_layer(inputs)
        env_enable_rat_embeddings = env_enable_embeddings * tf.expand_dims(
            rationale[:, :, 1], 2)

        # expand dim for envs
        max_seq_length = inputs.shape[1]
        envs_ = tf.tile(tf.expand_dims(envs, axis=1), [1, max_seq_length, 1])
        envs_ = tf.cast(envs_, tf.float32)
        envs_ = masks_ * envs_

        # concate the env ids to the embeddings
        env_enable_enc_inputs = tf.concat([env_enable_rat_embeddings, envs_],
                                          axis=-1)

        env_enable_enc_outputs = self.env_enable_encoder(env_enable_enc_inputs)

        # mask before max pooling
        # (1 - mask) * (-1e9) --> <pad> become very neg
        env_enable_enc_outputs_ = env_enable_enc_outputs * masks_ + (
            1. - masks_) * (-1e9)

        # aggregates hidden outputs via max pooling
        # shape -- (batch_size, hidden_dim * 2)
        env_enable_enc_output = tf.reduce_max(env_enable_enc_outputs_, axis=1)

        # task prediction
        # shape -- (batch_size, num_classes)
        env_enable_logits = self.env_enable_fc(env_enable_enc_output)

        return rationale, env_inv_logits, env_enable_logits

    def generator_trainable_variables(self):
        """
        Return a list of trainable variables of the generator.
        """
        variables = self.gen_embed_layer.trainable_variables
        variables += self.generator.trainable_variables
        variables += self.generator_fc.trainable_variables

        return variables

    def env_inv_trainable_variables(self):
        """
        Return a list of trainable variables of the 
        environment invariant predictor.
        """
        variables = self.env_inv_embed_layer.trainable_variables
        variables += self.env_inv_encoder.trainable_variables
        variables += self.env_inv_fc.trainable_variables

        return variables

    def env_enable_trainable_variables(self):
        """
        Return a list of trainable variables of the 
        environment enabled predictor.
        """
        variables = self.env_enable_embed_layer.trainable_variables
        variables += self.env_enable_encoder.trainable_variables
        variables += self.env_enable_fc.trainable_variables

        return variables
