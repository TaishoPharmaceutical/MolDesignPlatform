import tensorflow as tf

import time
import numpy as np
from tqdm import tqdm

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def scaled_dot_product_attention(q, k, v, mask, additional_weights=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    attention_weights = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        attention_weights += (mask * -1e9)

    attention_weights = tf.nn.softmax(attention_weights, axis=-1)

    if additional_weights is not None:
        attention_weights*=additional_weights

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, additional_weights=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if additional_weights is not None:
            additional_weights = additional_weights[:, tf.newaxis, :, :]
            additional_weidhts = tf.concat([additional_weights]*self.num_heads, 1)

        x, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, additional_weights)

        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.d_model))
        x = self.dense(x)

        return x, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
    ])



class GateLayer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()

        self.Wz = tf.keras.layers.Dense(d_model,use_bias=False)
        self.Wr = tf.keras.layers.Dense(d_model,use_bias=False)
        self.Wh = tf.keras.layers.Dense(d_model,use_bias=False)

        self.sigmoid = tf.keras.activations.sigmoid
        self.tanh = tf.keras.activations.tanh

    def call(self, h, x):

        z = self.sigmoid(self.Wz(x)+self.Wz(h))
        r = self.sigmoid(self.Wr(x)+self.Wr(h))
        hh = self.tanh(self.Wh(x)+self.Wh(h*r))

        h_next = (1-z) * h + z * hh

        return h_next


class GRUGateLayer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()

        self.Wxz = tf.keras.layers.Dense(d_model,use_bias=False)
        self.Whz = tf.keras.layers.Dense(d_model)
        self.Wxr = tf.keras.layers.Dense(d_model,use_bias=False)
        self.Whr = tf.keras.layers.Dense(d_model)
        self.Wxh = tf.keras.layers.Dense(d_model,use_bias=False)
        self.Whh = tf.keras.layers.Dense(d_model)

        self.sigmoid = tf.keras.activations.sigmoid
        self.tanh = tf.keras.activations.tanh

    def call(self, h, x):

        z = self.sigmoid(self.Wxz(x)+self.Whz(h))
        r = self.sigmoid(self.Wxr(x)+self.Whr(h))
        hh = self.tanh(self.Wxh(x)+self.Whh(h*r))

        h_next = (1-z) * h + z * hh

        return h_next


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gate = GateLayer(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, additional_weights):

        x_, _ = self.mha(x, x, x, mask, additional_weights)
        x_ = self.dropout1(x_, training=training)
        x = self.layernorm1(self.gate(x, x_))

        x_ = self.ffn(x)
        x_ = self.dropout2(x_, training=training)
        x = self.layernorm2(x + x_)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

        x_, block1 = self.mha1(x, x, x, look_ahead_mask)
        x_ = self.dropout1(x_, training=training)
        x = self.layernorm1(x_ + x)

        x_, block2 = self.mha2(
            enc_output, enc_output, x, padding_mask)
        x_ = self.dropout2(x_, training=training)
        x = self.layernorm2(x + x_)

        x_ = self.ffn(x)
        x_ = self.dropout3(x_, training=training)
        x = self.layernorm3(x + x_)

        return x, block1, block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1, use_embedding=False, use_pos_enc=False):
        super(Encoder, self).__init__()

        self.use_embedding=use_embedding
        self.use_pos_enc = use_pos_enc

        self.d_model = d_model
        self.num_layers = num_layers

        if use_embedding:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        if use_pos_enc:
            self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, additional_weights):

        seq_len = tf.shape(x)[1]

        if self.use_embedding:
            x = self.embedding(x)

        if self.use_pos_enc:
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask, additional_weights)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1, use_embedding=True):
        super(Decoder, self).__init__()
        self.use_embedding=use_embedding

        self.d_model = d_model
        self.num_layers = num_layers

        if use_embedding:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask, pos_enc=True):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        if self.use_embedding:
            x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if pos_enc:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights


class GatedGraphTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1, use_embedding=True, use_final_layer=True):
        super(GatedGraphTransformer, self).__init__()
        self.use_embedding=use_embedding
        self.use_final_layer=use_final_layer
        self.dense = tf.keras.layers.Dense(d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate, use_embedding)

        if use_final_layer:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, pos_enc=True):

        masks = create_masks(inp[2],tar)
        x = self.dense(inp[0])
        x = self.encoder(x, True, masks[0], inp[1])

        dec_output, attention_weights = self.decoder(
            tar, x, True, masks[1], masks[2], pos_enc)

        if self.use_final_layer:
            final_output = self.final_layer(dec_output)
        else:
            final_output = dec_output

        return final_output, attention_weights

    def reconstruct(self, inp, start_token, max_len, batch_size, multinomial=False):
        output = tf.ones([batch_size, 1], dtype=tf.int32)*start_token
        flags = None

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp[2], output)

        x = self.dense(inp[0])
        enc_output = self.encoder(x, False, enc_padding_mask, inp[1])

        for i in range(max_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp[2], output)

            predictions, attention_weights = self.decoder(
                output, enc_output, False, combined_mask, dec_padding_mask)

            predictions = self.final_layer(predictions)

            if multinomial:
                predictions = predictions[: ,-1, :]
                predicted_id = tf.compat.v1.multinomial(predictions, 1,output_dtype=tf.int32)
            else:
                predictions = predictions[: ,-1:, :]
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)


            f = tf.math.equal(predicted_id, start_token+1)
            if flags is None:
                flags = f.numpy()
            else:
                flags = flags | f.numpy()

            if np.sum(flags) == batch_size:
                return output, attention_weights

            output = tf.concat([output, predicted_id], axis=-1)

        return output, attention_weights

    def beam_search(self, inputs, start_token, end_token, width, length=100, alpha=0., full_batch=1000):
        output = tf.ones([1, 1], dtype=tf.int32)*start_token
        flag = None
        width = width
        score = tf.zeros([1])
        res = []
        res_score = []
        inp = inputs
        batch_size = tf.shape(output)[0]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inputs[2], output)

        x = self.dense(inputs[0])
        enc_output = self.encoder(x, False, enc_padding_mask, inputs[1])

        for i in tqdm(range(length)):

            if batch_size > full_batch:
                times = int(np.ceil(batch_size/full_batch))
                os = []
                ss = []

                for i in range(times):
                    bs = tf.shape(inp_s)[0]

                    inp_arr = tf.tile(inputs[2], [bs,1])
                    inp_s = tf.tile(enc_output, [bs, 1, 1])
                    out_s = output[i*full_batch:(i+1)*full_batch]
                    score_s = score[i*full_batch:(i+1)*full_batch]

                    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                        inp_arr, out_s)
                    predictions, attention_weights = self.decoder(
                        out_s, inp_s, False, combined_mask, dec_padding_mask)
                    predictions = self.final_layer(predictions)

                    out_s, score_s = BeamSearchCore(out_s, pred[: ,-1, :], bs, score_s, alpha=alpha)
                    os.append(out_s)
                    ss.append(score_s)

                output = tf.concat(os, 0)
                score = tf.concat(ss, 0)


            else:
                inp_arr = tf.tile(inputs[2], [batch_size,1])
                inp = tf.tile(enc_output, [batch_size, 1, 1])
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp_arr, output)
                predictions, attention_weights = self.decoder(
                    output, inp, False, combined_mask, dec_padding_mask)
                predictions = self.final_layer(predictions)

                predictions = predictions[: ,-1, :]  # (batch_size, 1, vocab_size)
                output, score = BeamSearchCore(output, predictions, width, score, alpha=alpha)

            last_token = output[:, -1]
            ends = tf.reshape(tf.math.equal(last_token, end_token), [-1])
            num_ends = tf.reduce_sum(tf.cast(ends, dtype=tf.int32))

            if num_ends != 0:
                res.append(tf.boolean_mask(output, ends))
                res_score.append(tf.boolean_mask(score, ends)/(i+1))

            output = tf.boolean_mask(output, ~ends)
            score = tf.boolean_mask(score, ~ends)

            width -= tf.reduce_sum(tf.cast(ends, dtype=tf.int32))

            batch_size = tf.shape(output)[0]

            if width == 0:
                return res, res_score

        return res, res_score


class GatedGraphTransformerS(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1, use_embedding=True, use_final_layer=True, latent_activation="sigmoid"):
        super(GatedGraphTransformerS, self).__init__()
        self.d_model=d_model
        self.use_embedding=use_embedding
        self.use_final_layer=use_final_layer
        self.dense = tf.keras.layers.Dense(d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        self.rv = tf.keras.layers.RepeatVector(pe_input)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate, use_embedding)

        self.sample=tf.keras.layers.Dense(d_model, activation=latent_activation)

        if use_final_layer:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, pos_enc=True):

        masks = create_masks(inp[2],tar)
        x = self.dense(inp[0])
        x = self.encoder(x, True, masks[0], inp[1])
        x = self.sample(x[:,0])
        x = self.rv(x)
        dec_output, attention_weights = self.decoder(
            tar, x, True, masks[1], masks[2], pos_enc)

        if self.use_final_layer:
            final_output = self.final_layer(dec_output)
        else:
            final_output = dec_output

        return final_output, attention_weights

    def average_pooling(self, x, mask):
        mask = tf.math.equal(mask[:,0,0,:], 0)
        mask1 = tf.cast(mask, dtype=tf.float32)
        mask2 = tf.reduce_sum(mask1, -1)[:, tf.newaxis]

        mask1 = tf.tile(mask1[:,:,tf.newaxis], tf.shape(x[:1,:1,:]))
        mask2 = tf.tile(mask2, tf.shape(x[:1,1,:]))

        x = tf.reduce_sum(x*mask1,1)
        x /= mask2

        return x

    def rsample(self, start_token, pe_input, batch_size=100, min_len=5, max_len=30):
        output = tf.ones([batch_size, 1], dtype=tf.int32)*start_token
        flags = None

        ones = tf.ones([pe_input,pe_input], dtype=tf.float32)
        ones = tf.compat.v1.matrix_band_part(ones, -1, 0)
        rand = tf.random.uniform([batch_size], minval=min_len, maxval=max_len, dtype=tf.int32)
        seq = tf.gather(ones, rand, axis=0)

        x = tf.abs(tf.random.normal([batch_size, self.d_model], stddev=1))
        g = tf.cast(tf.math.greater_equal(x, 0), dtype=tf.float32)
        l = tf.cast(tf.math.less(x, 0), dtype=tf.float32)
        x = x*g - tf.math.log(x)*l

        x = tf.keras.activations.sigmoid(x)
        enc_output = self.rv(x)

        for i in range(max_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                seq, output)

            predictions, attention_weights = self.decoder(
                output, enc_output, False, combined_mask, dec_padding_mask)

            predictions = self.final_layer(predictions)

            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            f = tf.math.equal(predicted_id, start_token+1)
            if flags is None:
                flags = f.numpy()
            else:
                flags = flags | f.numpy()

            if np.sum(flags) == batch_size:
                return output, attention_weights

            output = tf.concat([output, predicted_id], axis=-1)

        return output, attention_weights


    def reconstruct(self, inp, start_token, max_len, batch_size, multinomial=False, noise=0.):
        output = tf.ones([batch_size, 1], dtype=tf.int32)*start_token
        flags = None

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp[2], output)

        x = self.dense(inp[0])
        x = self.encoder(x, False, enc_padding_mask, inp[1])
        x = self.average_pooling(x, enc_padding_mask)

        x = self.sample(x)
        noise1 = tf.keras.activations.tanh(tf.random.normal(tf.shape(x), stddev=1)) * noise
        x = tf.abs(x+noise1)
        g = tf.cast(tf.math.greater(x, 1), dtype=tf.float32)
        x = x - (1-x)*g

        enc_output = self.rv(x)

        for i in range(max_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp[2], output)

            predictions, attention_weights = self.decoder(
                output, enc_output, False, combined_mask, dec_padding_mask)

            predictions = self.final_layer(predictions)

            if multinomial:
                predictions = predictions[: ,-1, :]  # (batch_size, 1, vocab_size)
                predicted_id = tf.compat.v1.multinomial(predictions, 1,output_dtype=tf.int32)
            else:
                predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            f = tf.math.equal(predicted_id, start_token+1)
            if flags is None:
                flags = f.numpy()
            else:
                flags = flags | f.numpy()

            if np.sum(flags) == batch_size:
                return output, attention_weights

            output = tf.concat([output, predicted_id], axis=-1)

        return output, attention_weights

    def beam_search(self, inputs, start_token, end_token, width, length=100, alpha=0., full_batch=1000):
        output = tf.ones([1, 1], dtype=tf.int32)*start_token
        flag = None
        width = width
        score = tf.zeros([1])
        res = []
        res_score = []
        inp = inputs
        batch_size = tf.shape(output)[0]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inputs[2], output)

        x = self.dense(inputs[0])
        x = self.encoder(x, False, enc_padding_mask, inputs[1])
        x = self.average_pooling(x, enc_padding_mask)
        x = self.sample(x)
        enc_output = self.rv(x)

        for i in tqdm(range(length)):

            if batch_size > full_batch:
                times = int(np.ceil(batch_size/full_batch))
                os = []
                ss = []

                for i in range(times):
                    bs = tf.shape(inp_s)[0]

                    inp_arr = tf.tile(inputs[2], [bs,1])
                    inp_s = tf.tile(enc_output, [bs, 1, 1])
                    out_s = output[i*full_batch:(i+1)*full_batch]
                    score_s = score[i*full_batch:(i+1)*full_batch]

                    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                        inp_arr, out_s)
                    predictions, attention_weights = self.decoder(
                        out_s, inp_s, False, combined_mask, dec_padding_mask)
                    predictions = self.final_layer(predictions)

                    out_s, score_s = BeamSearchCore(out_s, predictions[: ,-1, :], bs, score_s, alpha=alpha)
                    os.append(out_s)
                    ss.append(score_s)

                output = tf.concat(os, 0)
                score = tf.concat(ss, 0)

            else:
                inp_arr = tf.tile(inputs[2], [batch_size,1])
                inp = tf.tile(enc_output, [batch_size, 1, 1])
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp_arr, output)
                predictions, attention_weights = self.decoder(
                    output, inp, False, combined_mask, dec_padding_mask)
                predictions = self.final_layer(predictions)

                predictions = predictions[: ,-1, :]  # (batch_size, 1, vocab_size)
                output, score = BeamSearchCore(output, predictions, width, score, alpha=alpha)

            last_token = output[:, -1]
            ends = tf.reshape(tf.math.equal(last_token, end_token), [-1])
            num_ends = tf.reduce_sum(tf.cast(ends, dtype=tf.int32))

            if num_ends != 0:
                res.append(tf.boolean_mask(output, ends))
                res_score.append(tf.boolean_mask(score, ends)/(i+1))

            output = tf.boolean_mask(output, ~ends)
            score = tf.boolean_mask(score, ~ends)

            width -= tf.reduce_sum(tf.cast(ends, dtype=tf.int32))

            batch_size = tf.shape(output)[0]

            if width == 0:
                return res, res_score

        return res, res_score

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_en.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def calc_score(array, length, alpha=0.):
    return tf.math.log(tf.nn.softmax(array))/(((5. + tf.cast(length, dtype=tf.float32)) ** alpha) / ((5. + 1.) ** alpha))


def search(token, score, width):
    g = tf.math.top_k(score, k=width)[1]
    token = tf.gather(token, g)
    score = tf.gather(score, g)

    return token, score


def BeamSearchCore(token, array, width, score, alpha=0.):
    vocab_size = tf.shape(array)[-1]
    batch_size = tf.shape(array)[0]
    length = tf.shape(token)[-1]
    token = tf.reshape(tf.tile(token, [1, vocab_size]), [-1, length])
    nt = tf.tile(tf.reshape(tf.range(0, vocab_size, dtype=tf.int32), [-1,1]), [batch_size, 1])
    token = tf.concat([token,nt], -1)

    score = tf.reshape(score, [-1,1])
    score = tf.tile(score, [1, vocab_size])
    score += calc_score(array, length, alpha)
    score = tf.reshape(score, [-1])

    if width < tf.shape(token)[0]:
        token, score = search(token, score, width)

    return token, score


def BeamSearch(transformer, inputs, start_token, end_token, width, length=100, alpha=0., full_batch=1000):
    output = tf.ones([1, 1], dtype=tf.int32)*start_token
    flag = None
    width = width
    score = tf.zeros([1])
    res = []
    res_score = []
    inp = inputs
    batch_size = tf.shape(output)[0]

    for i in tqdm(range(length)):

        if batch_size > full_batch:
            times = int(np.ceil(batch_size/full_batch))
            os = []
            ss = []

            for i in range(times):
                inp_s = inp[i*full_batch:(i+1)*full_batch]
                out_s = output[i*full_batch:(i+1)*full_batch]
                score_s = score[i*full_batch:(i+1)*full_batch]

                bs = tf.shape(inp_s)[0]
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                    inp_s, out_s)
                pred, _= transformer(inp_s, out_s, False,enc_padding_mask,combined_mask,dec_padding_mask)
                out_s, score_s = BeamSearchCore(out_s, pred[: ,-1, :], bs, score_s, alpha=alpha)
                os.append(out_s)
                ss.append(score_s)

            output = tf.concat(os, 0)
            score = tf.concat(ss, 0)

        else:
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, output)

            predictions, _ = transformer(inp,
                                         output,
                                         False,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)

            predictions = predictions[: ,-1, :]
            output, score = BeamSearchCore(output, predictions, width, score, alpha=alpha)

        last_token = output[:, -1]
        ends = tf.reshape(tf.math.equal(last_token, end_token), [-1])
        num_ends = tf.reduce_sum(tf.cast(ends, dtype=tf.int32))

        if num_ends != 0:
            res.append(tf.boolean_mask(output, ends))
            res_score.append(tf.boolean_mask(score, ends)/(i+1))

        output = tf.boolean_mask(output, ~ends)
        score = tf.boolean_mask(score, ~ends)

        width -= tf.reduce_sum(tf.cast(ends, dtype=tf.int32))

        batch_size = tf.shape(output)[0]
        inp = tf.tile(inputs, [batch_size, 1])

        if width == 0:
            return res, res_score

    return res, res_score
