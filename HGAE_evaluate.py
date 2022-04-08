import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
from HGAE_model import HyperGraphAutoEncoder
from tqdm import tqdm
import numpy as np

events_per_batch = 100
num_batch = 1000


def read_tfrecord(example):
    feature_map = {
        'jet_feature': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    particles_features = tf.io.decode_raw(parsed_example['jet_feature'], out_type=tf.float64)
    particles_features = tf.cast(tf.reshape(particles_features, shape=(-1, 3)), tf.float32)
    num_particles = tf.shape(particles_features)[0]
    return particles_features, num_particles


def concat_jet(particles_features_batch, num_particles_batch):
    num_particles_max = tf.reduce_max(num_particles_batch)
    particles_features_concat = tf.concat([
        tf.split(particles_features_batch[i, :],
                 num_or_size_splits=[num_particles_batch[i], num_particles_max - num_particles_batch[i]],
                 axis=0)[0] for i in range(events_per_batch)], axis=0)
    return particles_features_concat, num_particles_batch


hyper_graph_auto_encoder_model = HyperGraphAutoEncoder(
    num_subgraph=1,
    batch_size=events_per_batch,
)
hyper_graph_auto_encoder_model.load_weights('hgae_weights/')


def get_error(tfrecord_file, file_name, num_total_event):
    dataset = tf.data.TFRecordDataset(tfrecord_file) \
        .map(map_func=read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .padded_batch(batch_size=events_per_batch) \
        .map(map_func=concat_jet, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .prefetch(tf.data.experimental.AUTOTUNE).take(num_batch)
    errors = np.zeros(shape=(num_total_event,))
    i = 0
    with tqdm(total=int(num_total_event / events_per_batch)) as pbar:
        for element in dataset.as_numpy_iterator():
            jets_feature_concat, num_particles = element
            event_reconstruct = hyper_graph_auto_encoder_model(
                [jets_feature_concat, tf.expand_dims(num_particles, axis=-1)])
            rmse = tf.sqrt(
                tf.math.segment_mean(tf.reduce_mean(tf.square(event_reconstruct - jets_feature_concat), axis=-1),
                                     tf.repeat(tf.cast(tf.range(events_per_batch), tf.int64), num_particles)))
            errors[i * events_per_batch: (i + 1) * events_per_batch, ] = tf.sqrt(rmse)
            pbar.update(1)
            i = i + 1
    np.save(file_name + '_error', errors)


# blackbox_1_tfrecord = "./dataset/events_LHCO2020_BlackBox1.tfrecord"
# blackbox_2_tfrecord = "./dataset/events_LHCO2020_BlackBox2.tfrecord"
# blackbox_3_tfrecord = "./dataset/events_LHCO2020_BlackBox3.tfrecord"
RnD_background_tfrecord = "./dataset/RnD_background.tfrecord"
RnD_signal_1_tfrecord = "./dataset/RnD_signal_1.tfrecord"
RnD_signal_2_tfrecord = "./dataset/RnD_signal_2.tfrecord"

get_error(RnD_background_tfrecord, file_name="./ResultEvaluate/RnD_background", num_total_event=100000)
get_error(RnD_signal_1_tfrecord, file_name="./ResultEvaluate/RnD_signal_1", num_total_event=100000)
get_error(RnD_signal_2_tfrecord, file_name="./ResultEvaluate/RnD_signal_2", num_total_event=100000)
# get_error(blackbox_1_tfrecord, file_name="blackbox_1_tfrecord", num_total_event=1000000)
# get_error(blackbox_2_tfrecord, file_name="blackbox_2_tfrecord", num_total_event=1000000)
# get_error(blackbox_3_tfrecord, file_name="blackbox_3_tfrecord", num_total_event=1000000)
