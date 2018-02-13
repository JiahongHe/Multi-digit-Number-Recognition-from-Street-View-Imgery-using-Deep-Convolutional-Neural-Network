import tensorflow as tf


class Model(object):
    # A Covolutional Neural Network model with 8 convolutional hidden layers and 3 dense layers.
    # the input:
    # input_x: input images.
    # drop_rate: drop rate set for Dropout Regularization.
    # filter_map: the number of filter used in convolutional layers.
    # kernel_size_map: the size of kernel used in convolutional layers and maxpooling layer.
    @staticmethod
    def layers (input_x, drop_rate,
                filter_map=[48,64,128,160,192],
                kernel_size_map=[[5,5],[2,2]],
                strides_map=[1,2]
                  ):
        # first 8 hidden layers are convolutional layers, all with batch normalization, ReLU activation, maxpooling and dropout regularization.
        
        # hidden layer 1
        conv_layer_1 = tf.layers.conv2d(input_x, filters=filter_map[0],
                                        kernel_size=kernel_size_map[0],
                                        padding='same')
        norm_layer_1 = tf.layers.batch_normalization(conv_layer_1)
        activation_layer_1 = tf.nn.relu(norm_layer_1)
        maxpooling_layer_1 = tf.layers.max_pooling2d(activation_layer_1,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[1],
                                                         padding='same')
        dropout_layer_1 = tf.layers.dropout(maxpooling_layer_1, rate=drop_rate)
        hidden_layer_1 = dropout_layer_1

        # hidden layer 2
        conv_layer_2 = tf.layers.conv2d(hidden_layer_1,
                                            filters=filter_map[1],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_2 = tf.layers.batch_normalization(conv_layer_2)
        activation_layer_2 = tf.nn.relu(norm_layer_2)
        maxpooling_layer_2 = tf.layers.max_pooling2d(activation_layer_2,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[0],
                                                         padding='same')
        dropout_layer_2 = tf.layers.dropout(maxpooling_layer_2, rate=drop_rate)
        hidden_layer_2 = dropout_layer_2

        # hidden layer 3
        conv_layer_3 = tf.layers.conv2d(hidden_layer_2,
                                            filters=filter_map[2],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_3 = tf.layers.batch_normalization(conv_layer_3)
        activation_layer_3 = tf.nn.relu(norm_layer_3)
        maxpooling_layer_3 = tf.layers.max_pooling2d(activation_layer_3,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[1],
                                                         padding='same')
        dropout_layer_3 = tf.layers.dropout(maxpooling_layer_3, rate=drop_rate)
        hidden_layer_3 = dropout_layer_3

        # hidden layer 4
        conv_layer_4 = tf.layers.conv2d(hidden_layer_3,
                                            filters=filter_map[3],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_4 = tf.layers.batch_normalization(conv_layer_4)
        activation_layer_4 = tf.nn.relu(norm_layer_4)
        maxpooling_layer_4 = tf.layers.max_pooling2d(activation_layer_4,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[0],
                                                         padding='same')
        dropout_layer_4 = tf.layers.dropout(maxpooling_layer_4, rate=drop_rate)
        hidden_layer_4 = dropout_layer_4

        # hidden layer 5
        conv_layer_5 = tf.layers.conv2d(hidden_layer_4,
                                            filters=filter_map[4],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_5 = tf.layers.batch_normalization(conv_layer_5)
        activation_layer_5 = tf.nn.relu(norm_layer_5)
        maxpooling_layer_5 = tf.layers.max_pooling2d(activation_layer_5,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[1],
                                                         padding='same')
        dropout_layer_5 = tf.layers.dropout(maxpooling_layer_5, rate=drop_rate)
        hidden_layer_5 = dropout_layer_5

        # hidden layer 6
        conv_layer_6 = tf.layers.conv2d(hidden_layer_5,
                                            filters=filter_map[4],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_6 = tf.layers.batch_normalization(conv_layer_6)
        activation_layer_6 = tf.nn.relu(norm_layer_6)
        maxpooling_layer_6 = tf.layers.max_pooling2d(activation_layer_6,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[0],
                                                         padding='same')
        dropout_layer_6 = tf.layers.dropout(maxpooling_layer_6, rate=drop_rate)
        hidden_layer_6 = dropout_layer_6

        # hidden layer 7
        conv_layer_7 = tf.layers.conv2d(hidden_layer_6,
                                            filters=filter_map[4],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_7 = tf.layers.batch_normalization(conv_layer_7)
        activation_layer_7 = tf.nn.relu(norm_layer_7)
        maxpooling_layer_7 = tf.layers.max_pooling2d(activation_layer_7,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[1],
                                                         padding='same')
        dropout_layer_7 = tf.layers.dropout(maxpooling_layer_7, rate=drop_rate)
        hidden_layer_7 = dropout_layer_7

        #hidden layer 8
        conv_layer_8 = tf.layers.conv2d(hidden_layer_7,
                                            filters=filter_map[4],
                                            kernel_size=kernel_size_map[0],
                                            padding='same')
        norm_layer_8 = tf.layers.batch_normalization(conv_layer_8)
        activation_layer_8 = tf.nn.relu(norm_layer_8)
        maxpooling_layer_8 = tf.layers.max_pooling2d(activation_layer_8,
                                                         pool_size=kernel_size_map[1],
                                                         strides=strides_map[0],
                                                         padding='same')
        dropout_layer_8 = tf.layers.dropout(maxpooling_layer_8, rate=drop_rate)
        hidden_layer_8 = dropout_layer_8

        flatten = tf.reshape(hidden_layer_8, [-1, 4 * 4 * 192])
        
        # 3 hidden dense layers

        # hidden layer 9
        dense = tf.layers.dense(flatten,
                                    units=3072,
                                    activation=tf.nn.relu)
        hidden_layer_9 = dense

        # hidden layer 10'
        dense = tf.layers.dense(hidden_layer_9,
                                    units=3072,
                                    activation=tf.nn.relu)
        hidden_layer_10 = dense

        # digit length
        dense = tf.layers.dense(hidden_layer_10, units=7)
        length = dense

        # digit1
        dense = tf.layers.dense(hidden_layer_10, units=11)
        digit1 = dense

        # digit2
        dense = tf.layers.dense(hidden_layer_10, units=11)
        digit2 = dense

        # digit3
        dense = tf.layers.dense(hidden_layer_10, units=11)
        digit3 = dense

        # digit4
        dense = tf.layers.dense(hidden_layer_10, units=11)
        digit4 = dense

        # digit5
        dense = tf.layers.dense(hidden_layer_10, units=11)
        digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logits, digits_logits
    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
