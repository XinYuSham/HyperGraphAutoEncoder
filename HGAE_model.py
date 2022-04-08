import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from SMH_VAE_load import dim_latent, dim_multi_hot, decoder_model


class MessageOperatorInput(layers.Layer):
    def __init__(
            self,
            num_edges,
            output_dim,
            input_dim,
            activation="relu",
            name=None,
            trainable=True
    ):
        super(MessageOperatorInput, self).__init__(name=name, trainable=trainable)
        self.num_edges = num_edges
        self.output_dim = output_dim
        self.activation = activation
        self.trainable = trainable
        self.activate_layer = layers.Activation(activation=activation)
        self.input_dim = input_dim
        self.edges_input_core = self.add_weight(
            shape=(num_edges, output_dim, input_dim),
            trainable=trainable,
            initializer="glorot_normal",
            name='edges_input_core'
        )
        self.edges_input_bias = self.add_weight(
            shape=(num_edges, output_dim, 1),
            trainable=trainable,
            initializer="zeros",
            name='edges_input_bias'
        )

    def call(self, inputs, *args, **kwargs):
        nodes_features_input, edges_index = inputs
        edges_input_core = tf.gather(self.edges_input_core, indices=edges_index)
        edges_input_bias = tf.gather(self.edges_input_bias, indices=edges_index)
        nodes_features_output = self.activate_layer(
            tf.matmul(edges_input_core, nodes_features_input) + edges_input_bias)
        return nodes_features_output


class MessageOperatorOutput(layers.Layer):
    def __init__(
            self,
            num_edges,
            batch_size,
            output_dim,
            input_dim,
            activation="relu",
            name=None,
            trainable=True
    ):
        super(MessageOperatorOutput, self).__init__(name=name, trainable=trainable)
        self.num_edges = num_edges
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.trainable = trainable
        self.activate_layer = layers.Activation(activation=activation)
        self.edges_output_core = self.add_weight(
            shape=(1, self.num_edges, self.output_dim, self.input_dim),
            trainable=self.trainable,
            initializer="glorot_normal",
            name='edges_output_core',
        )
        self.edges_output_bias = self.add_weight(
            shape=(1, self.num_edges, self.output_dim, 1),
            trainable=self.trainable,
            initializer="zeros",
            name='edges_output_bias'
        )

    def call(self, inputs, *args, **kwargs):
        edges_features_input = inputs
        edges_output_core = tf.reshape(tf.repeat(self.edges_output_core, repeats=self.batch_size, axis=0),
                                       (self.batch_size * self.num_edges, self.output_dim, self.input_dim))
        edges_output_bias = tf.reshape(tf.repeat(self.edges_output_bias, repeats=self.batch_size, axis=0),
                                       (self.batch_size * self.num_edges, self.output_dim, 1))
        edges_features_output = self.activate_layer(
            tf.matmul(edges_output_core, edges_features_input) + edges_output_bias)
        return edges_features_output


class EdgesFeatures(layers.Layer):
    def __init__(
            self,
            inputs_units,
            outputs_units,
            batch_size,
            num_subgraph
    ):
        super(EdgesFeatures, self).__init__()
        self.num_subgraph = num_subgraph
        self.batch_size = batch_size
        self.message_operator_inputs = [MessageOperatorInput(
            num_edges=num_subgraph * dim_multi_hot,
            output_dim=inputs_units[i + 1],
            input_dim=inputs_units[i],
        ) for i in range(len(inputs_units) - 1)]
        self.message_operator_outputs = [MessageOperatorOutput(
            num_edges=num_subgraph * dim_multi_hot,
            batch_size=batch_size,
            output_dim=outputs_units[i + 1],
            input_dim=outputs_units[i],
        ) for i in range(len(outputs_units) - 1)]

    def call(self, inputs, *args, **kwargs):
        nodes_features, incidence_matrix = inputs
        nodes_features_update = tf.gather(tf.expand_dims(nodes_features, axis=-1),
                                          incidence_matrix[:, 1] * self.num_subgraph
                                          + incidence_matrix[:, 2]
                                          )
        for i in range(len(self.message_operator_inputs)):
            nodes_features_update = self.message_operator_inputs[i]([nodes_features_update,
                                                                     incidence_matrix[:, 2] * dim_multi_hot
                                                                     + incidence_matrix[:, 3]
                                                                     ])
        edges_features = tf.math.unsorted_segment_mean(
            data=nodes_features_update,
            segment_ids=
            incidence_matrix[:, 0] * dim_multi_hot * self.num_subgraph
            + incidence_matrix[:, 2] * dim_multi_hot
            + incidence_matrix[:, 3],
            num_segments=self.batch_size * self.num_subgraph * dim_multi_hot
        )

        for j in range(len(self.message_operator_outputs)):
            edges_features = self.message_operator_outputs[j](edges_features)

        return edges_features


class NodesFeaturesToIncidenceMatrix(layers.Layer):
    def __init__(
            self,
            num_subgraph,
            batch_size,
            trainable=True
    ):
        super(NodesFeaturesToIncidenceMatrix, self).__init__(trainable=trainable)
        self.num_subgraph = num_subgraph
        self.batch_size = batch_size
        self.nodes_feature_to_multi_hot = [
            layers.Dense(units=512, activation="relu", trainable=trainable),
            layers.Dense(units=512, activation="relu", trainable=trainable),
            layers.Dense(units=512, activation="relu", trainable=trainable),
            layers.Dense(units=num_subgraph * dim_latent, trainable=trainable),
            layers.Reshape(target_shape=(num_subgraph, dim_latent))
        ]

    def call(self, inputs, *args, **kwargs):
        nodes_features, num_nodes = inputs
        nodes_features_multi_hot = nodes_features
        for i in range(len(self.nodes_feature_to_multi_hot)):
            nodes_features_multi_hot = self.nodes_feature_to_multi_hot[i](nodes_features_multi_hot)

        incidence_matrix = decoder_model(nodes_features_multi_hot)
        self.add_metric(tf.reduce_mean(tf.cast(incidence_matrix, tf.float32), axis=-1) * dim_multi_hot,
                        name="degree_mean")
        incidence_matrix = tf.where(tf.equal(incidence_matrix, 1))
        incidence_matrix_events_id = tf.gather(
            tf.repeat(tf.cast(tf.range(self.batch_size), tf.int64), num_nodes),
            incidence_matrix[:, 0]
        )
        incidence_matrix = tf.concat([tf.expand_dims(incidence_matrix_events_id, axis=-1), incidence_matrix], axis=-1)
        return incidence_matrix


class NodesFeaturesUpdate(layers.Layer):
    def __init__(
            self,
            batch_size,
            num_subgraph,
            edges_dim,
    ):
        super(NodesFeaturesUpdate, self).__init__()
        self.batch_size = batch_size
        self.num_subgraph = num_subgraph
        self.edges_dim = edges_dim

    def call(self, inputs, *args, **kwargs):
        edges_features, incidence_matrix, num_particles_total = inputs
        edges_features = tf.reshape(edges_features,
                                    (self.batch_size, self.num_subgraph * dim_multi_hot, self.edges_dim))
        edges_features = edges_features - tf.repeat(tf.expand_dims(tf.reduce_mean(edges_features, axis=0), axis=0),
                                                    self.batch_size, axis=0)
        edges_features = tf.reshape(edges_features,
                                    (self.batch_size * self.num_subgraph * dim_multi_hot, self.edges_dim))

        edges_features_gather = tf.gather(edges_features,
                                          incidence_matrix[:, 0] * self.num_subgraph * dim_multi_hot
                                          + incidence_matrix[:, 2] * dim_multi_hot
                                          + incidence_matrix[:, 3]
                                          )
        nodes_features_update = tf.math.unsorted_segment_sum(data=edges_features_gather,
                                                             segment_ids=incidence_matrix[:,
                                                                         1] * self.num_subgraph + incidence_matrix[:,
                                                                                                  2],
                                                             num_segments=num_particles_total * self.num_subgraph)
        return nodes_features_update


class HyperGraphAutoEncoder(keras.Model):
    def __init__(
            self,
            num_subgraph,
            batch_size,
    ):
        super(HyperGraphAutoEncoder, self).__init__()
        self.num_subgraph = num_subgraph
        self.batch_size = batch_size
        self.incidence_matrix_generatr = NodesFeaturesToIncidenceMatrix(
            num_subgraph=num_subgraph,
            batch_size=batch_size,
            trainable=True
        )
        self.edges_features = EdgesFeatures(inputs_units=[3, 8],
                                            outputs_units=[8, 8],
                                            batch_size=batch_size,
                                            num_subgraph=num_subgraph,
                                            )
        self.nodes_feature_update = NodesFeaturesUpdate(
            batch_size=batch_size,
            num_subgraph=num_subgraph,
            edges_dim=8
        )

        self.edges_features_1 = EdgesFeatures(inputs_units=[8, 8],
                                            outputs_units=[8, 8],
                                            batch_size=batch_size,
                                            num_subgraph=num_subgraph,
                                            )
        self.nodes_feature_update_1 = NodesFeaturesUpdate(
            batch_size=batch_size,
            num_subgraph=num_subgraph,
            edges_dim=8
        )

        self.particles_features_reconstruct = [
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(3)
        ]

    def call(self, inputs):
        particles_features, num_particles = inputs
        particles_features_repeat = tf.repeat(particles_features, self.num_subgraph, axis=0)
        num_particles_total = tf.reduce_sum(num_particles)
        num_particles = tf.squeeze(num_particles, axis=-1)

        incidence_matrix = self.incidence_matrix_generatr([particles_features, num_particles])

        edges_features = self.edges_features([particles_features_repeat, incidence_matrix])
        nodes_features_update = self.nodes_feature_update([edges_features, incidence_matrix, num_particles_total])

        edges_features = self.edges_features_1([nodes_features_update, incidence_matrix])
        nodes_features_update = self.nodes_feature_update_1([edges_features, incidence_matrix, num_particles_total])

        particles_features_reconstruct = tf.reshape(nodes_features_update, (num_particles_total, self.num_subgraph * 8))
        for i in range(len(self.particles_features_reconstruct)):
            particles_features_reconstruct = self.particles_features_reconstruct[i](particles_features_reconstruct)
        reconstruct_loss = tf.reduce_mean(tf.sqrt(tf.math.segment_mean(
            data=tf.reduce_mean(tf.square(particles_features_reconstruct - particles_features), axis=-1),
            segment_ids=tf.repeat(tf.range(self.batch_size), num_particles)
        )))
        self.add_loss(reconstruct_loss)
        self.add_metric(reconstruct_loss, "rmse")
        return particles_features_reconstruct
