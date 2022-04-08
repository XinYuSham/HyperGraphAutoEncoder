import tensorflow as tf
from SMH_VAE_model import decoder_model, dim_latent, dim_multi_hot, encoder_model

_ = decoder_model(tf.ones((1, dim_latent)))
# _ = encoder_model(tf.ones((1, dim_multi_hot)))
tf.train.Checkpoint(
    decoder_layer=decoder_model.decoder_layer
).restore(tf.train.latest_checkpoint("Decoder_weight/")).assert_consumed()

# tf.train.Checkpoint(
#     encoder_layer=encoder_model.encoder_layer
# ).restore(tf.train.latest_checkpoint("Encoder_weight/")).assert_consumed()
