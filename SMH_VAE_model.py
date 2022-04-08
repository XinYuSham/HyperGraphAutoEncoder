import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
dim_multi_hot = 64
dim_latent = 16


class Sampling(layers.Layer):
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class EncoderLayer(layers.Layer):
    def __init__(self,
                 multi_hot_dim,
                 latent_dim,
                 name=None):
        super(EncoderLayer, self).__init__(name=name)
        self.encoder_layers_mean = [
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=latent_dim),
        ]
        self.encoder_layers_log_var = [
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=latent_dim),
        ]

        self.sampling = Sampling()

    def call(self, inputs, **kwargs):
        multi_hot = inputs
        represent_mean = multi_hot
        represent_log_var = multi_hot
        for i in range(len(self.encoder_layers_mean)):
            represent_mean = self.encoder_layers_mean[i](represent_mean)
            represent_log_var = self.encoder_layers_log_var[i](represent_log_var)
        represent = self.sampling([represent_mean, represent_log_var])
        return represent_mean, represent_log_var, represent
        # return represent_mean
        # return multi_hot / (tf.reduce_max(multi_hot) - tf.reduce_min(multi_hot))


class DecoderLayer(layers.Layer):
    def __init__(self, multi_hot_dim,
                 latent_dim, name=None):
        super(DecoderLayer, self).__init__(name=name)
        self.decoder_layers = [
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=1024, activation="relu"),
            layers.Dense(units=1024, activation="relu"),
        ]
        self.reconstruct = layers.Dense(units=multi_hot_dim, activation="sigmoid")

    def call(self, inputs, **kwargs):
        multi_hot_reconstruct = inputs
        for i in range(len(self.decoder_layers)):
            multi_hot_reconstruct = self.decoder_layers[i](multi_hot_reconstruct)
        multi_hot_reconstruct = self.reconstruct(multi_hot_reconstruct)
        return multi_hot_reconstruct


class MultiHotVAE(keras.Model):
    def __init__(self,
                 multi_hot_dim,
                 latent_dim,
                 name=None,
                 ):
        super(MultiHotVAE, self).__init__(name=name)
        self.multi_hot_dim = multi_hot_dim
        self.latent_dim = latent_dim
        self.encoder = EncoderLayer(multi_hot_dim=multi_hot_dim, latent_dim=latent_dim)
        self.decoder = DecoderLayer(multi_hot_dim=multi_hot_dim, latent_dim=latent_dim)

    def call(self, inputs):
        multi_hot = tf.cast(inputs, tf.float32)
        represent_mean, represent_log_var, represent = self.encoder(multi_hot + 1e-2 * tf.keras.backend.random_uniform(shape=(tf.shape(multi_hot)[0], tf.shape(multi_hot)[1])))
        multi_hot_reconstruct = self.decoder(represent)
        diff = tf.math.unsorted_segment_mean(tf.reshape(tf.abs(multi_hot - multi_hot_reconstruct), (-1,)),
                                             tf.reshape(tf.cast(multi_hot, tf.int64), (-1,)),
                                             2)
        loss = tf.reduce_mean(diff) * self.multi_hot_dim
        self.add_loss(loss)

        alpha = self.latent_dim * 1e-1
        kl_loss = -5e-1 * tf.reduce_mean(
            1 + represent_log_var - tf.math.square(represent_mean) - tf.math.exp(represent_log_var)) * alpha
        self.add_loss(kl_loss)

        multi_hot_reconstruct = tf.cast(tf.less(0.5, multi_hot_reconstruct), tf.int32)
        reconstruct_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(tf.cast(multi_hot, tf.float32) - tf.cast(multi_hot_reconstruct, tf.float32)), axis=-1))
        self.add_metric(reconstruct_loss, "reconstruct_loss")
        multi_hot_mean = tf.reduce_sum(tf.cast(multi_hot_reconstruct, tf.float32), axis=-1)
        self.add_metric(multi_hot_mean, "multi_hot_mean")
        return multi_hot_reconstruct

    def get_config(self):
        return {
            'multi_hot_dim': self.multi_hot_dim,
            'latent_dim': self.latent_dim,
            'name': self.name
        }


class Decoder(keras.Model):
    def __init__(self, multi_hot_dim, latent_dim):
        super(Decoder, self).__init__()
        self.multi_hot_dim = multi_hot_dim
        self.latent_dim = latent_dim
        self.decoder_layer = DecoderLayer(multi_hot_dim=multi_hot_dim, latent_dim=latent_dim)

    def call(self, inputs):
        multi_hot = self.decoder_layer(inputs)
        multi_hot_reconstruct = tf.cast(tf.less(0.5, multi_hot), tf.int32)
        return multi_hot_reconstruct


class Encoder(keras.Model):
    def __init__(self, multi_hot_dim, latent_dim):
        super(Encoder, self).__init__()
        self.multi_hot_dim = multi_hot_dim
        self.latent_dim = latent_dim
        self.encoder_layer = EncoderLayer(multi_hot_dim=multi_hot_dim, latent_dim=latent_dim)

    def call(self, inputs):
        represent_mean, _, _ = self.encoder_layer(tf.cast(inputs, tf.float32))
        return represent_mean


vae_model = MultiHotVAE(multi_hot_dim=dim_multi_hot,
                        latent_dim=dim_latent,
                        name="multi_hot_vae")
decoder_model = Decoder(multi_hot_dim=dim_multi_hot, latent_dim=dim_latent)
encoder_model = Encoder(multi_hot_dim=dim_multi_hot, latent_dim=dim_latent)
