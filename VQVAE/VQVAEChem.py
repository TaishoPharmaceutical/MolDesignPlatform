import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from rdkit import Chem
import math

from ChemUtils import ConvMol_Sssr, create_batch_sssr, cmol_to_onehotvec, onehotvec_to_mol, onehotvec_to_mol_no_length
from ChemUtils import MAX_LEN, MAX_NB, MAX_RING_SIZE, SMILES_LENGTH, ATOM_LEN, BOND_LEN, char_len, atom_list, char_list, char_dict
from Transformer.Transformer import Transformer, create_look_ahead_mask, create_padding_mask, create_masks, CustomSchedule, Encoder

MAX_LEN=61
MAX_NB=9
MAX_RING_SIZE=9
SMILES_LENGTH = 150

encode_size = 128
hidden_size = 256
num_embeddings=4096
commitment_cost=0.05


def create_atom_mask(size, max_len=MAX_LEN):
    mask = tf.linalg.band_part(tf.ones((max_len, max_len)), -1, 0)[size-1]
    return mask


def create_bond_mask(size, max_len=MAX_LEN):
    mask = tf.linalg.band_part(tf.ones((max_len, max_len)), -1, 0)[:size]
    mask = tf.pad(mask, [[0,max_len-size],[0,0]])
    return mask  # (seq_len, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  #


def create_atom_masks(inp_len, batch_size, max_len=MAX_LEN):
    atom_masks = [create_atom_mask(inp_len[i]) for i in range(batch_size)]
    atom_masks = tf.stack(atom_masks)

    return atom_masks

def create_bond_masks(inp_len, batch_size, max_len=MAX_LEN):
    bond_masks = [create_bond_mask(inp_len[i]) for i in range(batch_size)]
    bond_masks = tf.stack(bond_masks)

    return bond_masks


def create_masks(inp_len, tar_len, batch_size, max_len = MAX_LEN, training=True):
    enc_padding_mask = [1-create_atom_mask(inp_len[i]) for i in range(batch_size)]
    enc_padding_mask = tf.stack(enc_padding_mask)[:, tf.newaxis, tf.newaxis, :]
    dec_padding_mask = [1-create_atom_mask(inp_len[i]) for i in range(batch_size)]
    dec_padding_mask = tf.stack(dec_padding_mask)[:, tf.newaxis, tf.newaxis, :]

    if training:
        look_ahead_mask = create_look_ahead_mask(max_len)
        dec_target_padding_mask = [1-create_atom_mask(tar_len[i]) for i in range(batch_size)]
    else:
        look_ahead_mask = create_look_ahead_mask(tar_len[0])
        dec_target_padding_mask = [1-create_atom_mask(tar_len[i], tar_len[i]) for i in range(batch_size)]
    dec_target_padding_mask = tf.stack(dec_target_padding_mask)[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class RelationalGraphconv(tf.keras.models.Model):
    def __init__(self,layer_size, activation="relu"):
        super(RelationalGraphconv,self).__init__()
        self.dense1 = [tf.keras.layers.Dense(layer_size, activation=activation) for x in range(5)]
        self.dense2 = tf.keras.layers.Dense(layer_size, activation=activation)
        self.layer_size = layer_size

    def call(self,inputs):
        atom_features = inputs[0]
        adj_list = inputs[1]

        afs=[]
        for i in range(5):
            af=self.dense1[i](atom_features)
            afs.append(af)

        afs = tf.concat(afs,0)

        zero_feature=tf.zeros(shape=[1,self.layer_size])
        atom_features_c=tf.concat([zero_feature,afs],axis=0)
        feat_gather=tf.gather(atom_features_c,adj_list)
        sum_feat=tf.reduce_sum(feat_gather,1)

        o_feat=self.dense2(atom_features)

        out_tensor=o_feat+sum_feat

        return out_tensor


class VQVAELayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='embedding',
                                  shape=(self.embedding_dim, self.num_embeddings),
                                  initializer=self.initializer,
                                  trainable=True)

        super(VQVAELayer, self).build(input_shape)

    def call(self, x, quantize_only=False):

        if quantize_only:
            return self.quantize(x)
        else:
            flat_inputs = K.reshape(x, (-1, self.embedding_dim))
            distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                         - 2 * K.dot(flat_inputs, self.w)
                         + K.sum(self.w ** 2, axis=0, keepdims=True))

            encoding_indices = K.argmax(-distances, axis=1)
            encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
            quantized = self.quantize(encoding_indices)

            return encoding_indices, quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices)


class Encoder1(tf.keras.models.Model):
    def __init__(self, encode_size=64, hidden_size=256, max_len=100, num_embeddings=128, commitment_cost=0.25, *kwargs):
        super().__init__(*kwargs)
        self.encode_size = encode_size
        self.max_len = max_len
        self.hidden_size = hidden_size

        #encoder
        self.gcn = [RelationalGraphconv(hidden_size) for x in range(3)]
        self.batchnorm = [tf.keras.layers.BatchNormalization() for x in range(3)]
        self.gcn_dense = tf.keras.layers.Dense(hidden_size, activation = "relu")

        self.vq = VQVAELayer(encode_size, num_embeddings, commitment_cost)
        self.pre_conv1d = tf.keras.layers.Conv1D(encode_size, 1, padding='same')
        self.post_conv1d = tf.keras.layers.Conv1D(encode_size, 1, padding='same', activation='relu')
        self.batchnorm0 = tf.keras.layers.BatchNormalization()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        gcn_feat = inputs[0]
        adj_list = inputs[1]
        membership  = inputs[8]

        #eoncoder gcn
        x = gcn_feat
        for i in range(3):
            x = self.gcn[i]([x, adj_list])
            x = self.batchnorm[i](x)
        #x = Graphpool(x, pool_adj_list)

        #x = self.gcn_dense(x)
        zeros = tf.zeros_like(tf.expand_dims(x[0], 0), dtype=tf.float32)
        feat = tf.concat([zeros, x], 0)
        x = tf.gather(feat,membership)
        x = tf.stack(x)
        #x = self.max_pool3(x)

        qi = self.pre_conv1d(x)
        qi = self.batchnorm1(qi)
        indices, q= self.vq(qi)
        x = qi + K.stop_gradient(q-qi)

        return qi, q, indices, x

class Encoder2(tf.keras.models.Model):
    def __init__(self, encode_size=64, hidden_size=256, max_len=100, num_embeddings=128, commitment_cost=0.25, *kwargs):
        super().__init__(*kwargs)
        self.encode_size = encode_size
        self.max_len = max_len
        self.hidden_size = hidden_size

        #encoder
        self.gcn = [RelationalGraphconv(hidden_size) for x in range(3)]
        self.batchnorm = [tf.keras.layers.BatchNormalization() for x in range(3)]
        self.gcn_dense = tf.keras.layers.Dense(hidden_size, activation = "relu")

        self.sssr_gcn = [RelationalGraphconv(hidden_size) for x in range(1)]
        self.sssr_gcn_batchnorm = [tf.keras.layers.BatchNormalization() for x in range(1)]
        self.sssr_dense = tf.keras.layers.Dense(hidden_size, activation = "relu")
        self.self_dense1 = tf.keras.layers.Dense(hidden_size, activation = "relu")
        self.self_dense2 = tf.keras.layers.Dense(hidden_size, activation = "relu")
        self.sssr_batchnorm0 = tf.keras.layers.BatchNormalization()
        self.sssr_batchnorm1 = tf.keras.layers.BatchNormalization()
        self.self_batchnorm0 = tf.keras.layers.BatchNormalization()
        self.self_batchnorm1 = tf.keras.layers.BatchNormalization()
        self.gcn_dense2 = tf.keras.layers.Dense(hidden_size, activation = "relu")

        #self.conv1d_6 = tf.keras.layers.Conv1D(hidden_size, 6, padding='same', activation='relu')
        self.conv1d_2_1 = tf.keras.layers.Conv1D(hidden_size, 2, padding='same', activation='relu')
        self.conv1d_3 = tf.keras.layers.Conv1D(hidden_size, 3, padding='same', activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPool1D(3)
        #self.conv1d_3_2 = tf.keras.layers.Conv1D(hidden_size, 3, padding='same', activation='relu')
        self.conv1d_2_2 = tf.keras.layers.Conv1D(hidden_size, 3, padding='same', activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPool1D(2)
        #self.max_pool3 = tf.keras.layers.MaxPool1D(2)

        #self.gru_cell = tf.keras.layers.GRUCell(hidden_size)
        #self.gru = tf.keras.layers.RNN(self.gru_cell)
        #self.bigru = tf.keras.layers.Bidirectional(self.gru)

        self.vq = VQVAELayer(encode_size, num_embeddings, commitment_cost)
        self.pre_conv1d = tf.keras.layers.Dense(encode_size)
        #self.post_conv1d = tf.keras.layers.Conv1D(encode_size, 1, padding='same', activation='relu')
        self.batchnorm0 = tf.keras.layers.BatchNormalization()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        gcn_feat = inputs[0]
        adj_list = inputs[1]
        membership  = inputs[2]
        sssr = inputs[3]

        #eoncoder gcn
        x = gcn_feat
        for i in range(3):
            x = self.gcn[i]([x, adj_list])
            x = self.batchnorm[i](x)
        #x = Graphpool(x, pool_adj_list)

        #x = self.gcn_dense(x)
        zeros = tf.zeros_like(tf.expand_dims(x[0], 0), dtype=tf.float32)
        feat = tf.concat([zeros, x], 0)

        sssr_feat = tf.gather(feat, sssr)
        sssr_feat = self.conv1d_2_1(sssr_feat)
        sssr_feat = self.sssr_batchnorm0(sssr_feat)
        sssr_feat = self.conv1d_3(sssr_feat)
        sssr_feat = self.max_pool1(sssr_feat)
        sssr_feat = self.conv1d_2_2(sssr_feat)
        sssr_feat = self.max_pool2(sssr_feat)
        sssr_feat = tf.reshape(sssr_feat, [-1, self.hidden_size * int(int(MAX_RING_SIZE/3)/2)])
        #mask = tf.not_equal(sssr, 0)
        #sssr_feat = self.gru(sssr_feat, mask=mask)

        sssr_feat = self.sssr_dense(sssr_feat)
        sssr_feat = self.sssr_batchnorm1(sssr_feat)
        self_feat = self.self_dense1(x)
        self_feat = self.self_batchnorm0(self_feat)
        self_feat = self.self_dense2(self_feat)
        self_feat = self.self_batchnorm1(self_feat)

        x = tf.concat([self_feat, sssr_feat],0)

        #for i in range(1):
            #x = self.sssr_gcn[i]([x,adj_sssr])
            #x = self.sssr_gcn_batchnorm[i](x)
            #x = Graphpool(x, pool_adj_sssr)

        #x = self.gcn_dense2(x)
        zeros = tf.zeros_like(tf.expand_dims(x[0], 0), dtype=tf.float32)
        x = tf.concat([zeros, x], 0)
        x = self.pre_conv1d(x)
        #x = self.batchnorm1(x)

        x = tf.gather(x,membership)
        qi = tf.stack(x)
        #x = self.max_pool3(x)
        indices, q= self.vq(qi)
        x = qi + K.stop_gradient(q-qi)

        return qi, q, indices, x

class Decoder1(tf.keras.models.Model):
    def __init__(self, max_len=100, *kwargs):
        super().__init__(*kwargs)
        self.max_len = max_len

        self.dec_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dec_batchnorm = tf.keras.layers.BatchNormalization()
        self.dec_dense2 = tf.keras.layers.Dense(max_len*50, activation='relu')
        self.dec_batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dec_dense3 = tf.keras.layers.Dense(max_len*64, activation='relu')
        self.dec_batchnorm3 = tf.keras.layers.BatchNormalization()
        self.dec_dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dec_batchnorm4 = tf.keras.layers.BatchNormalization()

        self.dec_dense2_1 = tf.keras.layers.Dense(ATOM_LEN)
        self.dec_dense2_2 = tf.keras.layers.Dense(max_len*BOND_LEN)
        self.dec_dense2_3 = tf.keras.layers.Dense(max_len)
        self.softmax1 = tf.keras.layers.Softmax()
        self.softmax2 = tf.keras.layers.Softmax()
        self.softmax3 = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.dec_dense1(inputs)
        x = self.dec_batchnorm(x)
        x = self.dec_dense2(x)
        x = self.dec_batchnorm2(x)

        x = self.dec_dense3(x)
        x = self.dec_batchnorm3(x)
        x = tf.reshape(x, [-1, 64])
        x = self.dec_dense4(x)
        x = self.dec_batchnorm4(x)

        atom_feat = self.dec_dense2_1(x)
        bond_feat = self.dec_dense2_2(x)
        atom_feat = tf.reshape(atom_feat, [-1, self.max_len, ATOM_LEN])
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len, self.max_len, BOND_LEN])
        atom_feat = self.softmax1(atom_feat)
        bond_feat = self.softmax2(bond_feat)

        return atom_feat, bond_feat

class RGCDecoderPlus(tf.keras.models.Model):
    def __init__(self, encode_size, max_len, hidden_size, *kwargs):
        super().__init__(*kwargs)
        self.max_len = max_len

        self.rv = tf.keras.layers.RepeatVector(max_len)
        self.gru_celld0 = tf.keras.layers.GRUCell(hidden_size)
        self.grud0 = tf.keras.layers.RNN(self.gru_celld0,return_sequences=True)
        self.gru_celld1 = tf.keras.layers.GRUCell(hidden_size)
        self.grud1 = tf.keras.layers.RNN(self.gru_celld1,return_sequences=True)
        self.bigrud = tf.keras.layers.Bidirectional(self.grud1)
        self.dense1=tf.keras.layers.Dense(hidden_size)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.conv1d_1 = tf.keras.layers.Conv1D(ATOM_LEN, 1)
        self.conv1d_2 = tf.keras.layers.Conv1D(max_len * BOND_LEN, 1)
        self.dec_dense2_3 = tf.keras.layers.Dense(max_len)
        self.softmax1 = tf.keras.layers.Softmax()
        self.softmax2 = tf.keras.layers.Softmax()
        self.softmax3 = tf.keras.layers.Softmax()

    def call(self, inputs):

        x = self.dense1(inputs)
        x = self.batchnorm1(x)
        length = self.dec_dense2_3(x)
        length = self.softmax3(length)

        x = self.rv(x)
        x = self.grud0(x)
        x = self.batchnorm2(x)
        x = self.bigrud(x)
        x = self.batchnorm3(x)
        #pad = tf.constant([[0,0],[6,6],[0,0]], dtype=tf.int32)
        #x = tf.pad(x, pad)

        atom_feat = self.conv1d_1(x)
        bond_feat = self.conv1d_2(x)
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len, self.max_len, BOND_LEN])
        atom_feat = self.softmax1(atom_feat)
        bond_feat = self.softmax2(bond_feat)

        return atom_feat, bond_feat


class TDecoder(tf.keras.models.Model):
    def __init__(self, max_len, num_layers=4,d_model=encode_size,num_heads=4,dff=128,
                 input_vocab_size=num_embeddings, target_vocab_size=num_embeddings,
                 pe_input=num_embeddings, pe_target=num_embeddings,rate=0.1,*kwargs):
        super().__init__(*kwargs)
        self.max_len = max_len
        self.transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size,
                                       target_vocab_size, pe_input, pe_target, rate, False, False)
        self.dense=tf.keras.layers.Dense(d_model)
        self.dense_a = tf.keras.layers.Dense(ATOM_LEN)
        self.softmax_a = tf.keras.layers.Softmax()
        self.dense_b = tf.keras.layers.Dense(max_len * BOND_LEN)
        self.softmax_b = tf.keras.layers.Softmax()
        self.dense_c = tf.keras.layers.Dense(max_len)
        self.softmax_c = tf.keras.layers.Softmax()

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        zeros = tf.zeros_like(inp[:, :1], dtype=tf.float32)
        inp = tf.concat([zeros, inp], 1)[:,:-1]

        atom_feat = tar[:, :self.max_len*ATOM_LEN]
        bond_feat = tar[:, self.max_len*ATOM_LEN:]

        atom_feat = tf.reshape(atom_feat, [-1, self.max_len, ATOM_LEN])
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len, self.max_len*BOND_LEN])
        tar = tf.concat([atom_feat, bond_feat], axis=-1)

        zeros = tf.zeros_like(tar[:, :1], dtype=tf.float32)
        tar = tf.concat([zeros, tar], 1)[:,:-1]
        tar = self.dense(tar)

        x, attention_weights = self.transformer(inp, tar, training, enc_padding_mask,
                                               look_ahead_mask, dec_padding_mask)

        atom_feat = self.dense_a(x)
        bond_feat = self.dense_b(x)
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len, self.max_len, BOND_LEN])
        atom_feat = self.softmax_a(atom_feat)
        bond_feat = self.softmax_b(bond_feat)

        return atom_feat, bond_feat


    def decode(self, inp, batch_size, inp_len):

        zeros = tf.zeros_like(inp[:, :1], dtype=tf.float32)
        inp = tf.concat([zeros, inp], 1)[:,:-1]
        tar = tf.zeros([batch_size, 1, self.max_len*BOND_LEN + ATOM_LEN], dtype=tf.float32)
        inp_len = inp_len+1
        tar_len = tf.ones([batch_size], dtype=tf.int32)
        masks = create_masks(inp_len, tar_len, batch_size, self.max_len, False)
        encodes = self.transformer.encoder(inp, False, masks[0])

        for i in range(self.max_len):
            tar_in = self.dense(tar)
            tar_len = tf.ones([batch_size], dtype=tf.int32)+i
            masks = create_masks(inp_len, tar_len, batch_size, self.max_len, False)
            x, attention_weights = self.transformer.decoder(tar_in, encodes, False,
                                                            masks[1], masks[2])

            atom_feat = self.dense_a(x)
            bond_feat = self.dense_b(x)
            bond_feat = tf.reshape(bond_feat, [-1, i+1, self.max_len, BOND_LEN])
            atom_feat = self.softmax_a(atom_feat)
            bond_feat = self.softmax_b(bond_feat)

            af = tf.argmax(atom_feat, axis=-1)
            atom_feat = tf.one_hot(af, ATOM_LEN, dtype=tf.float32)
            bond_feat = tf.argmax(bond_feat, axis=-1)
            bond_feat = tf.one_hot(bond_feat, BOND_LEN, dtype=tf.float32)
            bond_feat = tf.reshape(bond_feat, [-1, i+1, self.max_len*BOND_LEN])
            feats = tf.concat([atom_feat, bond_feat], axis = -1)
            feats = feats[:,-1:, :]

            if tf.reduce_sum(af) == 0 & i < self.max_len-1:
                atom_feat = tf.concat([atom_feat,tf.zeros([batch_size, self.max_len-i-1, ATOM_LEN])], axis=1)
                bond_feat = tf.concat([atom_feat,tf.zeros([batch_size, self.max_len-i-1, self.max_len*BOND_LEN])], axis=1)
                break

            tar = tf.concat([tar, feats], axis = 1)

        return atom_feat, bond_feat


class VQVAE(tf.keras.models.Model):
    def __init__(self, encode_size=64, hidden_size=256, max_len=100, num_embeddings=128, commitment_cost=0.25,
                 num_layers=4,d_model=encode_size,num_heads=8,dff=512,
                 input_vocab_size=num_embeddings, target_vocab_size=num_embeddings,
                 pe_input=num_embeddings, pe_target=num_embeddings,rate=0.1,*kwargs):
        super().__init__(*kwargs)
        self.encode_size=encode_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.l_log = math.log(max_len)
        self.a_log = math.log(ATOM_LEN)
        self.b_log = math.log(BOND_LEN)
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.code_list = None
        self.code_dict = None

        self.encoder = Encoder2(encode_size, hidden_size, max_len, num_embeddings, commitment_cost)

        #decoder
        #self.decoder = Decoder1(max_len)
        self.tdecoder = TDecoder(max_len, num_layers=num_layers,d_model=d_model,num_heads=num_heads,dff=dff,
                                input_vocab_size=num_embeddings, target_vocab_size=num_embeddings,
                                pe_input=num_embeddings, pe_target=num_embeddings,rate=0.1)
        #self.decoder_plus = RGCDecoderPlus(encode_size, max_len, hidden_size*8)

        #vqdecoder
        #self.vqdecoder = VQDecoder(encode_size, max_len, hidden_size*8)

        #vae
        self.vae = VAE(encode_size, hidden_size, max_len, num_embeddings)


    def call(self, inputs, masks):
        qi, q, indices, x = self.encoder(inputs)
        vq_loss = self.vq_loss(q,qi, self.commitment_cost)
        atom_feat, bond_feat = self.tdecoder(x, inputs[5], True,
                                             masks[0], masks[1], masks[2])
        r_loss = self.restoration_loss(inputs, atom_feat, bond_feat)

        return r_loss, vq_loss, atom_feat, bond_feat

    def vae_train(self, inputs):
        r_loss, kl_loss, re_indices = self.vae(inputs)
        return r_loss, kl_loss, re_indices


    def encode(self, inputs):
        qi, q, indices, x = self.encoder(inputs)

        return indices, x


    def decode(self, inputs, batch_size, inputs_length):

        atom_feat, bond_feat = self.tdecoder.decode(inputs, batch_size, inputs_length)

        atom_feat = tf.reshape(atom_feat, [-1, self.max_len*ATOM_LEN])
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len*self.max_len*BOND_LEN])

        feats = tf.concat([atom_feat, bond_feat], -1)

        return feats


    def encode_from_smiles(self, smiles):
        mols = [Chem.MolFromSmiles(x) for x in smiles]

        for i in range(len(mols)):
            Chem.Kekulize(mols[i])

        cmols = [ConvMol_Sssr(x) for x in mols]
        feats = create_batch_sssr(cmols, max_len=self.max_len)

        qi, q, indices, x = self.encoder(feats)
        pre_latent_vector, latent_vector = self.vae.encode(indices, True)

        return use_smiles, indices, x, pre_latent_vector, latent_vector


    def rsample(self, z_vecs):
        z_mean = self.z_mean(z_vecs)
        z_log_var = -tf.abs(self.z_log_var(z_vecs))
        z_shape= tf.shape(z_mean)
        kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum(1.0 + z_log_var - z_mean*z_mean-tf.exp(z_log_var), -1))
        epsilon = tf.random.normal(z_shape)

        z_vecs = z_mean + tf.exp(z_log_var/2) * epsilon

        return z_vecs, kl_loss


    def sampling(self, n, e, zero_code = None, random='normal', z_mean = False, max_size= None, fix_indices = None):
        re_indices = self.vae.sampling(n,e, random, z_mean)
        re_indices = re_indices.numpy()
        if zero_code is None:
            zero_code = re_indices[0][-1]

        cindices = np.zeros_like(re_indices)+zero_code
        for i,indices in enumerate(re_indices):
            zero_pos = np.where(indices == zero_code)
            if len(zero_pos[0]) == 0:
                zero_pos = self.max_len
            else:
                zero_pos = zero_pos[0][0]
            cindices[i][0:zero_pos] = indices[0:zero_pos]
        re_indices = cindices

        if fix_indices is not None:
            cindices = np.zeros_like(re_indices)+zero_code
            for i,indices in enumerate(re_indices):
                ind = fix_indices + list(indices)
                zero_pos = np.where(indices == zero_code)
                if len(zero_pos[0]) == 0:
                    zero_pos = self.max_len
                else:
                    zero_pos = zero_pos[0][0]

                if zero_pos > self.max_len:
                    zero_pos = self.max_len
                cindices[i][0:zero_pos] = ind[0:zero_pos]
            re_indices = cindices

        if max_size is not None:
            cindices = np.zeros_like(re_indices)+zero_code
            for i,indices in enumerate(re_indices):
                cindices[i][0:max_size] = indices[0:max_size]
            re_indices = cindices

        batch_size = len(re_indices)
        inp_len = tf.cast(tf.not_equal(re_indices, zero_code), dtype=tf.int32)
        inp_len = tf.reduce_sum(inp_len, -1)+2

        x = self.encoder.vq(re_indices, True)
        x = self.decode(x,batch_size, inp_len)

        mols = []
        for y in x.numpy():
            try:
                mols.append(onehotvec_to_mol_no_length(y, MAX_LEN))
            except:
                continue

        return mols, re_indices


    def input_smiles_base_sampling(self, smiles, n=1, e=0.5, max_size= None, fix_indices = None, input_type='smiles', unique=True):

        if input_type=='smiles':
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol)
            cmol = ConvMol_Sssr(mol)
            feat = create_batch_sssr([cmol], max_len=self.max_len)
            qi, q, oindices, x = self.encoder(feat)
        else:
            oindices = np.reshape(smiles, [1,-1])

        zero_code = oindices[0][-1]
        x = self.vae.encode(oindices)
        x = tf.concat([x]*n, axis =0)
        noise = tf.random.normal([n, self.encode_size], 0)*e
        x += noise
        indices = self.vae.decode(x)

        if max_size is not None:
            indices = indices.numpy()
            cindices = np.zeros_like(indices)+indices[0][-1]
            for i,ind in enumerate(indices):
                cindices[i][0:max_size] = ind[0:max_size]
            indices = tf.convert_to_tensor(cindices, dtype=tf.int64)

        if fix_indices is not None:
            fix_indices = tf.convert_to_tensor(fix_indices, dtype=tf.int64)
            fix_indices = tf.one_hot(fix_indices, self.max_len, dtype = tf.int64)
            fix_indices = tf.reduce_sum(fix_indices, 0)
            indices = indices * (tf.ones_like(fix_indices,dtype=tf.int64)-fix_indices)
            indices+= oindices * fix_indices

        if unique:
            indices = np.unique(indices.numpy(), axis=0)

        indices = tf.convert_to_tensor(indices)

        batch_size = len(indices)
        inp_len = tf.cast(tf.not_equal(indices, zero_code), dtype=tf.int32)
        inp_len = tf.reduce_sum(inp_len, -1)+2

        x = self.encoder.vq(indices, True)
        x = self.decode(x,batch_size, inp_len)

        mols = [onehotvec_to_mol_no_length(y, MAX_LEN) for y in x.numpy()]

        return mols,indices


    def restoration(self, smiles):
        batch_size = len(smiles)
        mols = [Chem.MolFromSmiles(x) for x in smiles]

        for i in range(len(mols)):
            Chem.Kekulize(mols[i])

        cmols = [ConvMol_Sssr(x) for x in mols]
        feats = create_batch_sssr(cmols, max_len = self.max_len)
        length = np.array([x.num_nodes for x in cmols], dtype=np.int32)

        inp_len = tf.cast(tf.not_equal(feats[2], 0), dtype=tf.int32)
        inp_len = tf.reduce_sum(inp_len, -1)+2

        qi, q, indices, x = self.encoder(feats)
        x = self.decode(x, batch_size, inp_len)

        mols = [onehotvec_to_mol_no_length(y, MAX_LEN) for y in x.numpy()]

        return mols, indices


    def restoration_loss(self, inputs, atom_feat, bond_feat):
        batch_size = tf.shape(inputs[5])[0]

        mask = tf.cast(inputs[6], dtype=tf.int32)
        mask_ = tf.reduce_sum(mask, -1)
        mask_ = tf.one_hot(mask_, self.max_len, dtype=tf.int32)
        mask = tf.cast(mask+mask_, dtype=tf.bool)

        #true_length = inputs[9][:, 0:self.max_len]
        true_atom_feat = inputs[5][:, :self.max_len*ATOM_LEN]
        true_bond_feat = inputs[5][:, self.max_len*ATOM_LEN:]

        true_atom_feat = tf.reshape(true_atom_feat, [-1, self.max_len, ATOM_LEN])
        true_atom_feat = tf.boolean_mask(true_atom_feat, mask)
        atom_feat = tf.boolean_mask(atom_feat, mask)

        true_bond_feat = tf.reshape(true_bond_feat, [-1, self.max_len, self.max_len*BOND_LEN])
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len, self.max_len*BOND_LEN])
        true_bond_feat = tf.boolean_mask(true_bond_feat, mask)
        bond_feat = tf.boolean_mask(bond_feat, mask)
        true_bond_feat = tf.reshape(true_bond_feat, [-1, self.max_len, BOND_LEN])
        bond_feat = tf.reshape(bond_feat, [-1, self.max_len, BOND_LEN])

        #length_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(true_length, length))/self.l_log
        atom_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(true_atom_feat, atom_feat))
        bond_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(true_bond_feat, bond_feat))

        loss =  atom_loss + bond_loss

        return loss/tf.cast(batch_size, tf.float32)/1600


    def vq_loss(self, quantized, x_inputs, commitment_cost=0.25):

        e_latent_loss = K.mean((K.stop_gradient(quantized)-x_inputs)**2)
        q_latent_loss = K.mean((quantized-K.stop_gradient(x_inputs))**2)
        loss = q_latent_loss + commitment_cost * e_latent_loss

        return loss**0.5


class VAE(tf.keras.models.Model):
    def __init__(self, encode_size=64, hidden_size=256, max_len=100, num_embeddings=128, *kwargs):
        super().__init__(*kwargs)
        self.encode_size=encode_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.l_log = math.log(max_len)
        self.a_log = math.log(ATOM_LEN)
        self.b_log = math.log(BOND_LEN)
        self.num_embeddings = num_embeddings

        self.z_mean = tf.keras.layers.Dense(encode_size)
        self.z_log_var = tf.keras.layers.Dense(encode_size)

        self.embedding = tf.keras.layers.Embedding(num_embeddings,64, input_length = max_len)
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')


        self.batchnorm0 = tf.keras.layers.BatchNormalization()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()


        self.rv = tf.keras.layers.RepeatVector(max_len)
        self.gru_celld0 = tf.keras.layers.GRUCell(hidden_size)
        self.grud0 = tf.keras.layers.RNN(self.gru_celld0, return_sequences=True)
        self.gru_celld1 = tf.keras.layers.GRUCell(hidden_size)
        self.grud1 = tf.keras.layers.RNN(self.gru_celld1, return_sequences=True)
        self.bigrud =tf.keras.layers.Bidirectional(self.grud1)
        self.dense2 = tf.keras.layers.Dense(num_embeddings)
        self.softmax1 = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reshape(x, [-1, self.max_len*64])
        x = self.dense1(x)
        x = self.batchnorm0(x)

        z_vecs, kl_loss = self.rsample(x)

        x = self.rv(z_vecs)
        x = self.grud0(x)
        x = self.batchnorm1(x)
        x = self.bigrud(x)
        x = self.batchnorm2(x)
        x = self.dense2(x)
        x = self.softmax1(x)

        one_hot = tf.one_hot(inputs, self.num_embeddings)
        r_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(one_hot, x))

        re_indices = tf.argmax(x, -1)

        return r_loss, kl_loss, re_indices

    def encode(self, inputs, get_vecs=False):
        x = self.embedding(inputs)
        x = tf.reshape(x, [-1, self.max_len*64])
        x = self.dense1(x)
        x = self.batchnorm0(x)

        z_vecs  = self.z_mean(x)

        if get_vecs:
            return x, z_vecs

        return z_vecs

    def decode(self, inputs):
        x = self.rv(inputs)
        x = self.grud0(x)
        x = self.batchnorm1(x)
        x = self.bigrud(x)
        x = self.batchnorm2(x)
        x = self.dense2(x)
        x = self.softmax1(x)

        re_indices = tf.argmax(x, -1)

        return re_indices


    def rsample(self, z_vecs):
        z_mean = self.z_mean(z_vecs)
        z_log_var = -tf.abs(self.z_log_var(z_vecs))
        z_shape= tf.shape(z_mean)
        kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum(1.0 + z_log_var - z_mean*z_mean-tf.exp(z_log_var), -1))
        epsilon = tf.random.normal(z_shape)

        z_vecs = z_mean + tf.exp(z_log_var/2) * epsilon

        return z_vecs, kl_loss


    def sampling(self, n, e, random='normal', z_mean = False):
        if z_mean:
            if random == 'normal':
                x = tf.random.normal([n, self.encode_size])*e
                x,loss = self.rsample(x)
            else:
                x = (tf.random.uniform([n, self.encode_size])-0.5)*e
                x,loss = self.rsample(x)
        else:
            if random == 'normal':
                x = tf.random.normal([n, self.encode_size])*e
            else:
                x = (tf.random.uniform([n, self.encode_size])-0.5)*e

        x = self.rv(x)
        x = self.grud0(x)
        x = self.batchnorm1(x)
        x = self.bigrud(x)
        x = self.batchnorm2(x)
        x = self.dense2(x)
        x = self.softmax1(x)

        re_indices = tf.argmax(x, -1)

        return re_indices


def fit_compound(smiles, model, cycle=None, learning_rate = 0.00003, max_len=MAX_LEN, max_train=100):
    mol = Chem.MolFromSmiles(smiles)
    can_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    Chem.Kekulize(mol)
    cmols = [ConvMol_Sssr(mol)]

    gcn_inputs = create_batch_sssr(cmols, max_len=max_len)
    mf_batch = [cmol_to_onehotvec(x, max_len) for x in cmols]
    mf_batch = list(map(list, zip(*mf_batch)))

    feat = [gcn_inputs[0],gcn_inputs[1],gcn_inputs[2],gcn_inputs[3], gcn_inputs[4],
            mf_batch[0],mf_batch[1],mf_batch[2],mf_batch[3]]

    feat = [tf.convert_to_tensor(x) for x in feat]

    model.encoder.trainable=False
    model.vae.trainable=False
    model.tdecoder.transformer.trainable=False

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for i in range(max_train):
        with tf.GradientTape() as tape:
            inp_len = tf.cast(tf.not_equal(feat[2], 0), dtype=tf.int32)
            inp_len = tf.reduce_sum(inp_len, -1)+2
            tar_len = feat[-1]+1
            masks = create_masks(inp_len, tar_len, 1)
            r_loss, vq_loss, atom_feat, bond_feat = model(feat, masks)
            loss = r_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(r_loss)

        if cycle is not None:
            if i == cycle:
                break

        re_mols, indices = model.restoration([smiles])
        if re_mols[0] is not None:
            re_mols = Chem.MolFromSmiles(Chem.MolToSmiles(re_mols[0], isomericSmiles=False))
            if re_mols is not None:
                re_smiles = Chem.MolToSmiles(re_mols, isomericSmiles=False)
                if can_smiles == re_smiles:
                    break
