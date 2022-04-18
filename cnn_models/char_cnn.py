import tensorflow as tf


class CharCNN(object):
    def __init__(self, alphabet_size, document_max_len, num_class):
        self.learning_rate = 1e-4
        self.filter_sizes = [7, 7, 3, 3, 3, 3]
        self.num_filters = 256
        self.kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05)

        self.x = tf.compat.v1.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.compat.v1.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.compat.v1.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.compat.v1.where(self.is_training, 0.5, 1.0)

        self.x_one_hot = tf.one_hot(self.x, alphabet_size)
        self.x_expanded = tf.expand_dims(self.x_one_hot, -1)

        # ============= Convolutional Layers =============
        with tf.compat.v1.name_scope("conv-maxpool-1"):
            conv1 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[0], alphabet_size],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)(self.x_expanded)
            pool1 = tf.keras.layers.MaxPooling2D(
                pool_size=(3, 1),
                strides=(3, 1))(conv1)
            pool1 = tf.transpose(a=pool1, perm=[0, 1, 3, 2])

        with tf.compat.v1.name_scope("conv-maxpool-2"):
            conv2 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[1], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)(pool1)
            pool2 = tf.keras.layers.MaxPooling2D(
                pool_size=(3, 1),
                strides=(3, 1))(conv2)
            pool2 = tf.transpose(a=pool2, perm=[0, 1, 3, 2])

        with tf.compat.v1.name_scope("conv-3"):
            conv3 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[2], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)(pool2)
            conv3 = tf.transpose(a=conv3, perm=[0, 1, 3, 2])

        with tf.compat.v1.name_scope("conv-4"):
            conv4 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[3], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)(conv3)
            conv4 = tf.transpose(a=conv4, perm=[0, 1, 3, 2])

        with tf.compat.v1.name_scope("conv-5"):
            conv5 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[4], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)(conv4)
            conv5 = tf.transpose(a=conv5, perm=[0, 1, 3, 2])

        with tf.compat.v1.name_scope("conv-maxpool-6"):
            conv6 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[5], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)(conv5)
            pool6 = tf.keras.layers.MaxPooling2D(
                pool_size=(3, 1),
                strides=(3, 1))(conv6)
            pool6 = tf.transpose(a=pool6, perm=[0, 2, 1, 3])
            h_pool = tf.reshape(pool6, [-1, (1) * self.num_filters])  # x * 27 + 96

        # ============= Fully Connected Layers =============
        with tf.compat.v1.name_scope("fc-1"):
            fc1_out = tf.keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)(h_pool)

        with tf.compat.v1.name_scope("fc-2"):
            fc2_out = tf.keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)(fc1_out)

        with tf.compat.v1.name_scope("fc-3"):
            self.logits = tf.keras.layers.Dense(num_class, activation=None, kernel_initializer=self.kernel_initializer)(fc2_out)
            self.predictions = tf.argmax(input=self.logits, axis=-1, output_type=tf.int32)

        # ============= Loss and Accuracy =============
        with tf.compat.v1.name_scope("loss"):
            self.y_one_hot = tf.one_hot(self.y, num_class)
            self.loss = tf.reduce_mean(
                input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_one_hot))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.compat.v1.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_predictions, "float"), name="accuracy")

