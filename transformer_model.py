# imports
import common_functions
from data_generator import base_data_gen

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K


# positional embedding layer
# num_vocab - len of vocab, maxlen - len of input sequence, num_hid - num of output dims
class TokenEmbedding(layers.Layer):
    def __init__(self, maxlen, num_hid, **kwargs):
        super().__init__()
        self.maxlen = maxlen
        self.num_hid = num_hid
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "num_hid": self.num_hid,
        })
        return config


# speech embedding (downscale speech feature)
# num_hid - number of filters in conv1d
class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid, **kwargs):
        self.num_hid = num_hid
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid * 3, 11, padding="same", activation="selu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid * 2, 11, padding="same", activation="selu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid * 1, 11, padding="same", activation="selu"
        )

    def call(self, x):
        #
        #x = self.conv1(x)
        #x = self.conv2(x)
        return self.conv3(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hid": self.num_hid,
        })
        return config


# transformer encoder layer
# embed_dim - dim of output, num_heads - number of attention heads, feed_forward_dim - dim of dense, rate - dropout rate
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, head_size, feed_forward_dim, rate, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = head_size
        self.feed_forward_dim = feed_forward_dim
        self.rate = rate

        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="selu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "feed_forward_dim": self.feed_forward_dim,
            "rate": self.rate
        })
        return config


# ctc loss layer
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        #
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        #
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        #
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        #
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        #
        self.add_loss(loss)
        #
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update({
            "name": self.name,
        })
        return config


# now build the model
# transformer_layers_number - number of transformer layers
# t_head_size, t_num_heads, t_ff_dim, t_dropout - transformer layers params
# reshape_form - form of reshaped cnn output
# element_size - size of one element (in dense)
# softmax_len - len of dim in output softmax layer
# c_dropout, f_dropout - dropout in cnn part, dropout before final softmax layer
# img_h, img_w, img_c - params of input image
def build_model(transformer_layers_number, t_input_len, t_embedding_dim, t_head_size, t_num_heads, t_ff_dim, t_dropout, reshape_form,
                element_size, softmax_len, c_dropout, f_dropout, learning_rate, img_h, img_w, img_c):
    # image input
    input_img = layers.Input(shape=(img_h, img_w, img_c), dtype="float32", name="input_image")
    # true labels for loss input (only for train)
    labels = layers.Input(shape=(None,), dtype="float32", name="output_text")

    # first conv block
    x = layers.Conv2D(32, (3, 3), activation="selu", kernel_initializer="he_normal", padding="same")(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="selu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(c_dropout)(x)

    # second conv block
    x = layers.Conv2D(64, (3, 3), activation="selu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="selu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(c_dropout)(x)

    # third conv block
    x = layers.Conv2D(128, (3, 3), activation="selu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="selu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(c_dropout)(x)

    # split image to many element
    x = layers.Reshape(target_shape=reshape_form)(x) # 131072
    # processing all element with conv1d to reduce their size
    x = SpeechFeatureEmbedding(num_hid=element_size)(x)
    x = layers.LayerNormalization()(x)
    # positional embedding
    x = TokenEmbedding(maxlen=t_input_len, num_hid=t_embedding_dim)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(c_dropout)(x)

    # transformer block
    for layer in range(transformer_layers_number):
        x = TransformerEncoder(embed_dim=t_embedding_dim, num_heads=t_num_heads, head_size=t_head_size, feed_forward_dim=t_ff_dim, rate=t_dropout)(x)

    # output softmax layer
    x = layers.Dropout(f_dropout)(x)
    x = layers.TimeDistributed(layers.Dense(softmax_len, activation="softmax", name="label_output"))(x)

    # use ctc loss
    output = CTCLayer(name="ctc_loss")(labels, x)

    # build model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="transformer_model_v999")

    # compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    # return model
    return model

