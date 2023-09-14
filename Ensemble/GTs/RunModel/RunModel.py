import tensorflow as tf
from ..Models.GatedGraphTransformer import create_masks, CustomSchedule, Encoder
from ..Models.GraphTransformer import Encoder as Enc
from ..Models.Transformer import Decoder
tf.random.set_seed(123)

class PredictModel(tf.keras.models.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, reg_tasks, class_tasks, rate=0.1, encoder_type="normal"):
        super().__init__()
        self.reg_tasks=reg_tasks
        self.class_tasks=class_tasks
        self.d_model = d_model
        self.encoder_type=encoder_type

        self.dense = tf.keras.layers.Dense(d_model)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=rate, use_embedding=False, use_pos_enc=False)

        if encoder_type != "normal":
            self.decoder = Decoder(2,d_model, num_heads, dff, reg_tasks+class_tasks+1,
                               70, rate, use_embedding=True)

        self.reg_output=[tf.keras.layers.Dense(1) for x in range(reg_tasks)]
        self.class_output=[tf.keras.layers.Dense(2) for x in range(class_tasks)]
        self.activations = [tf.keras.layers.Softmax() for x in range(class_tasks)]


    def network(self, x, training, mask, additional_weights):
        if self.encoder_type != "normal":
            additional_weights = tf.concat([additional_weights]*4, 1)
        else:
            additional_weights = tf.concat([additional_weights[:,:1]]*20, 1)
        x = self.dense(x)
        x = self.encoder(x, training, mask, additional_weights)
        #x = self.average_pooling(x,mask)

        return x

    def call(self, x, dec_inp, training, mask, dmask, additional_weights):
        x = self.network(x, training, mask, additional_weights)

        if self.encoder_type != "normal":
            x, _ = self.decoder(dec_inp, x, training, dmask, mask)
            output1 = [tf.squeeze(self.reg_output[i](x[:,i]), -1) for i in range(self.reg_tasks)]
            output2 = [self.activations[i](self.class_output[i](x[:,i+self.reg_tasks])) for i in range(self.class_tasks)]
        else:
            x = self.average_pooling(x,mask)
            output1 = [tf.squeeze(self.reg_output[i](x), -1) for i in range(self.reg_tasks)]
            output2 = [self.activations[i](self.class_output[i](x)) for i in range(self.class_tasks)]

        return tf.stack(output1), tf.stack(output2)

    def get_value(self, x, dec_inp, training, mask, dmask, additional_weights):
        x = self.network(x, training, mask, additional_weights)

        if self.encoder_type != "normal":
            x, _ = self.decoder(dec_inp, x, training, dmask, mask)
            output1 = [tf.squeeze(self.reg_output[i](x[:,i]), -1) for i in range(self.reg_tasks)]
            output2 = [self.activations[i](self.class_output[i](x[:,i+self.reg_tasks])) for i in range(self.class_tasks)]
        else:
            x = self.average_pooling(x,mask)
            output1 = [tf.squeeze(self.reg_output[i](x), -1) for i in range(self.reg_tasks)]
            output2 = [self.activations[i](self.class_output[i](x)) for i in range(self.class_tasks)]

        return tf.stack(output1), tf.stack(output2)

    def get_weight(self, x, dec_inp, training, mask, dmask, additional_weights, max_len):
        x = self.network(x, training, mask, additional_weights)

        result2 = []
        x_ = self.average_pooling(x,mask)
        for i in range(self.class_tasks):
            result2.append(self.activations[i](self.class_output[i](x_)))

        result2 = tf.stack(result2)[:,:,1:2]
        result2 = tf.concat([result2]*max_len, -1)

        weight1 =[]
        for i in range(self.reg_tasks):
            weight1.append(tf.squeeze(self.reg_output[i](x),-1))
        weight1 = tf.stack(weight1)

        weight2 = []
        for i in range(self.class_tasks):
            weight2.append(self.class_output[i](x))
        weight2 = tf.stack(weight2)[:,:,:,1]
        weight2 = weight2*result2

        weight = tf.concat([weight1,weight2],0)

        return weight


    def finetuning_core(self, x, dec_inp, training, mask, dmask, additional_weights):
        x = self.network(x, training, mask, additional_weights)
        x, _ = self.decoder(dec_inp, x, training, dmask, mask)

        return x


    def finetuning(self, x, label, num_layer):

        #calc loss
        x = self.reg_output[num_layer](x)
        loss = self.regression_loss(label, x)

        return loss


    def average_pooling(self, x, mask):
        #batch_size=tf.shape(x)[0]
        mask = tf.math.equal(mask[:,0,0,:], 0)
        mask1 = tf.cast(mask, dtype=tf.float32)
        mask2 = tf.reduce_sum(mask1, -1)[:, tf.newaxis]

        mask1 = tf.tile(mask1[:,:,tf.newaxis], tf.shape(x[:1,:1,:]))
        mask2 = tf.tile(mask2, tf.shape(x[:1,1,:]))

        x = tf.reduce_sum(x*mask1,1)
        x /= mask2

        return x


    def max_pooling(self, x, mask):
        #batch_size=tf.shape(x)[0]
        mask = tf.math.not_equal(mask[:,0,0,:], 0)
        mask1 = tf.cast(mask, dtype=tf.float32)
        mask1 = -tf.tile(mask1[:,:,tf.newaxis], tf.shape(x[:1,:1,:]))*10**2

        x = tf.reduce_max(x+mask1,1)

        return x


    def regression_loss(self, label, prediction):
        bool_nan = tf.math.is_nan(label)
        label = tf.boolean_mask(label,~bool_nan)

        prediction = tf.boolean_mask(prediction,~bool_nan)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(label, prediction))

        return loss


    def classification_loss(self, label, prediction):
        bool_nan = tf.math.is_nan(label)
        label = tf.cast(tf.boolean_mask(label,~bool_nan), tf.int32)
        label = tf.one_hot(label, 2)

        prediction = tf.boolean_mask(prediction,~bool_nan)
        loss =tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label,prediction))

        return loss


    def loss_function(self, output1, output2, label1, label2):
        if self.reg_tasks == 0:
            loss = self.classification_loss(label2,output2)
        elif self.class_tasks == 0:
            loss = self.regression_loss(label1, output1)
        else:
            loss = self.regression_loss(label1, output1) + self.classification_loss(label2,output2)

        return loss


    def training(self, x, dec_inp, training, mask, dmask, additional_weights, label1, label2):
        output1, output2 = self.call(x, dec_inp, training, mask, dmask, additional_weights)
        loss = self.loss_function(output1, output2, label1, label2)

        return loss
