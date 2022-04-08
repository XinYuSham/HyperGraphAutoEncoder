import tensorflow as tf
from tensorflow.keras import optimizers
from HGAE_model import HyperGraphAutoEncoder
from CosineAnnealing import CosineAnnealing

dataset_tfrecord = './dataset/RnD_background.tfrecord'
total_events = 1000000
events_per_batch = 100
num_batch_total = total_events / events_per_batch
num_batch_train = int(0.7 * num_batch_total)
num_batch_valid = int(0.2 * num_batch_total)
num_batch_test = int(0.1 * num_batch_total)


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
    return (particles_features_concat, num_particles_batch), 0


dataset = tf.data.TFRecordDataset(dataset_tfrecord) \
    .map(map_func=read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .padded_batch(batch_size=events_per_batch) \
    .map(map_func=concat_jet, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

train_valid_dataset = dataset.skip(num_batch_test)
train_dataset = train_valid_dataset.take(num_batch_train)
valid_dataset = train_valid_dataset.skip(num_batch_train)

hyper_graph_auto_encoder_model = HyperGraphAutoEncoder(
    num_subgraph=1,
    batch_size=events_per_batch,
)

# hyper_graph_auto_encoder_model.load_weights("hgae_weights/")
hyper_graph_auto_encoder_model.compile(optimizer=optimizers.Adam(1e-4))
hyper_graph_auto_encoder_model.fit(train_dataset, validation_data=valid_dataset, epochs=1)
# epochs = [4]
# for Ti in epochs:
#     reduce_lr = CosineAnnealing(eta_max=1, eta_min=0, total_iteration=Ti * (7000 // 100), iteration=0, verbose=0)
#     hist1 = hyper_graph_auto_encoder_model.fit(train_dataset, validation_data=valid_dataset, epochs=Ti, callbacks=[reduce_lr])

# data = train_valid_dataset.as_numpy_iterator().next()
# a = hyper_graph_auto_encoder_model.particle_feature_to_incidence_matrix([data[0][0], data[0][1]])
hyper_graph_auto_encoder_model.save_weights("hgae_weights/")
