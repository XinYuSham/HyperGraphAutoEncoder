import random
import tensorflow as tf
from SMH_VAE_model import vae_model, dim_multi_hot
from tensorflow.keras import optimizers

batch_size = 10000
batch_num = 500000


def gen():
    for _ in range(batch_num):
        data = tf.random.uniform(shape=(batch_size, dim_multi_hot))
        data_arg_sort = tf.argsort(data, axis=-1)[:, 2:6]
        data_arg_sort_min = tf.reduce_min(data_arg_sort, axis=-1)
        data_compare = tf.expand_dims(tf.gather_nd(data, tf.stack([tf.range(batch_size), data_arg_sort_min], axis=-1)), axis=-1)
        yield tf.cast(tf.less_equal(data, data_compare), tf.int32)


dataset = tf.data.Dataset.from_generator(generator=gen, output_types=tf.int32,
                                         output_shapes=(batch_size, dim_multi_hot))
vae_model.compile(optimizer=optimizers.Adam(1e-4))
# vae_model.load_weights("VAE_weights/")
vae_model.fit(dataset)
vae_model.save_weights("VAE_weights/")

decoder_check_point = tf.train.Checkpoint(
    decoder_layer=vae_model.decoder,
)
encoder_check_point = tf.train.Checkpoint(
    encoder_layer=vae_model.encoder,
)
decoder_weight_path = decoder_check_point.save("Decoder_weight/")
encoder_weight_path = encoder_check_point.save("Encoder_weight/")
