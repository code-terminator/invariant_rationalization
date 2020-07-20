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


class Attention(tf.keras.layers.Layer):
    """Computing the attention over the inputs."""

    def __init__(self, proj_dim):
        super(Attention, self).__init__()

        self.proj_dim = proj_dim

        self.head = tf.keras.layers.Dense(units=1, use_bias=False)
        self.proj = tf.keras.layers.Dense(units=self.proj_dim,
                                          activation="tanh")

    def call(self, inputs, masks):
        """
        Inputs:
            inputs -- (batch_size, seq_length, input_dim)
            masks -- (batch_size, seq_length, 1)
        Outputs:
            output -- (batch_size, input_dim)        
            atts -- (batch_size, seq_length)
        """
        proj_inputs = self.proj(inputs)
        att_logits = self.head(proj_inputs)  # (batch_size, seq_length, 1)

        # <pad> should have attention score 0
        neg_inf = tf.ones(tf.shape(masks)) * (-1e9)
        att_logits = att_logits * masks + (1. - masks) * neg_inf

        atts = tf.nn.softmax(att_logits, axis=1)  # (batch_size, seq_length, 1)
        atts = atts * masks

        context_vecs = inputs * atts  # (batch_size, seq_length, input_dim)
        output = tf.reduce_sum(context_vecs, axis=1)

        return output, tf.squeeze(atts, axis=-1)


class InvRNNwithSpanPred(tf.keras.Model):
    """
    A RNN-based invariant rationalization model with 
    span prediction.
    """

    def __init__(self, args):
        super(InvRNNwithSpanPred, self).__init__()

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

        # generator attention layer (single head contrastive to ouput the start token)
        self.attn_layer = Attention(proj_dim=args.rnn_dim * 2)

        # encoder output layer (classification task)
        self.env_inv_fc = tf.keras.layers.Dense(units=args.num_classes)
        self.env_enable_fc = tf.keras.layers.Dense(units=args.num_classes)

    def _contrastive_straight_through_sampling(self, att):
        """
        Straight through via contrastive sampling.
        Inputs:
            atts -- shape (batch_size, sequence_length)
        Outputs:
            att_max -- the hard sampled atts with straight through gradient  
        """
        att_max = tf.one_hot(tf.math.argmax(att, axis=-1), att.shape[1])

        att_max = tf.stop_gradient(att_max - att) + att

        return att_max

    def _head2rationale(self, head_idx, k):
        """
        Use convolution operation to convert head idx 
        to rationale format (batch_size, sequence_length, 1)
        
        Inputs:
            head_idx: (batch_size, sequence_length, 1).
            k: the lenght of rationale. 
        Tips:
            use convolution with all ones, kernels size (k-1) * 2 + 1            
            
        Output:
            z : (batch_size, sequence_length, 1) 
            if value == 1, it is selected as rationale.              
        """
        kernels = tf.concat([
            tf.ones([k, 1, 1], dtype=tf.float32),
            tf.zeros([k - 1, 1, 1], dtype=tf.float32)
        ],
                            axis=0)

        z = tf.nn.conv1d(head_idx, kernels, 1, "SAME")

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

        ############## generator ##############
        gen_embeddings = masks_ * self.gen_embed_layer(inputs)
        gen_outputs = self.generator(gen_embeddings)

        ############## attentions ##############
        _, soft_atts = self.attn_layer(gen_outputs, masks_)

        head_idx = self._contrastive_straight_through_sampling(soft_atts)
        head_idx = tf.expand_dims(head_idx, -1)

        # use convolution to generate rationale (batch_size, sequence_length, 1)
        rationale = self._head2rationale(head_idx, self.args.rationale_length)

        # mask the rationale that corresponding to <pad>
        rationale = masks_ * rationale

        ############## env inv predictor ##############
        env_inv_embeddings = masks_ * self.env_inv_embed_layer(inputs)
        env_inv_rat_embeddings = env_inv_embeddings * rationale

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
        env_enable_rat_embeddings = env_enable_embeddings * rationale

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

        # rearrange output format
        # binarize
        rationale_ = tf.cast(tf.math.less(tf.math.abs(rationale - 1.), 1e-4),
                             rationale.dtype)
        rationale_ = tf.concat([1. - rationale_, rationale_], axis=-1)

        return rationale_, env_inv_logits, env_enable_logits

    def generator_trainable_variables(self):
        """
        Return a list of trainable variables of the generator.
        """
        variables = self.gen_embed_layer.trainable_variables
        variables += self.generator.trainable_variables
        variables += self.attn_layer.trainable_variables

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
