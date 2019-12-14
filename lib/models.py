'''
Module for Q function approximators
'''

import tensorflow as tf



class CNN(tf.keras.Model):
    '''
    Parameters:
    -----------
    embed_dim : integer
        dimension of embedding space in which each piece id is represented
    num_filters : integer
        number of filters in convolutional layer
    kernel_size : tuple
        kernel size for convolitional layer
    regularization_constant : float
        lambda for l2 regularization in fully connected layer, used to mitigate
        overfitting to replay buffer during each episode of training

    Attributes (Needs Updating):
    -----------
    embedding : tf.keras.layers.Embedding
        embedding layer to map each cube piece id
    cnn : tf.keras.layers.Conv3d
        convolutional layer
    flatten : tf.keras.layers.Flatten
        flatten convolution output into one long vector
    fc : tf.keras.layers.Dense
        fully connected layer w/ output dimension equal to number of cube moves
    lr : tf.keras.layers.LeakyReLU
        activation layer for self.fc, chosen to avoid dying relu issue

    '''
    def __init__(self,
                 embed_dim=50,
                 num_filters=20,
                 num_conv_layers=1,
                 kernel_size=2,
                 regularization_constant=0,
                 num_dense_layers=1,
                 dense_activation='elu',
                 conv_activation='elu'):
        super(CNN, self).__init__()
        assert num_conv_layers > 0, "at least 1 conv layer required"
        assert num_dense_layers > 0, "at least 1 dense layer required"


        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.rc = regularization_constant
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.embedding = tf.keras.layers.Embedding(27, self.embed_dim)
        self.cnn = self._build_conv_layers(num_conv_layers,
                                           activation=self.conv_activation)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = self._build_fc_layers(num_dense_layers,
                                        activation=self.dense_activation)
        self.output_layer = tf.keras.layers.Dense(12,
                                            activation=self.dense_activation,
                                            kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                            bias_regularizer=tf.keras.regularizers.l2(l=self.rc))

    def _build_conv_layers(self, num_conv_layers, activation=None):
        '''
        helper function for init
        '''
        cnn = tf.keras.Sequential()
        for i in range(num_conv_layers):
            if i == 0:
                padding = 'valid'
            else:
                padding = 'same'
            cnn_layer = tf.keras.layers.Conv3D(filters=self.num_filters,
                                               kernel_size=self.kernel_size,
                                               data_format='channels_last',
                                               padding = padding,
                                               activation=activation)
            cnn.add(cnn_layer)
        return cnn


    def _build_fc_layers(self, num_dense_layers, activation=None):
        '''
        helper function for init
        '''
        fc = tf.keras.Sequential()
        for i in range(num_dense_layers):
            fc_layer = tf.keras.layers.Dense(50,
                                             kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                             bias_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                             activation=activation)
            fc.add(fc_layer)
        return fc


    def call(self, x, training=True):
        '''
        Forward pass for CNN

        Parameters:
        -----------
        x : tensorflow.python.framework.ops.EagerTensor
            shape is (batch_size, 3, 3, 3) representing rubix cube
        '''
        x = self.embedding(x)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.output_layer(x)
        return x



class CNNWithBatchNorm(tf.keras.Model):
    '''
    Parameters:
    -----------
    embed_dim : integer
        dimension of embedding space in which each piece id is represented
    num_filters : integer
        number of filters in convolutional layer
    kernel_size : tuple
        kernel size for convolitional layer
    regularization_constant : float
        lambda for l2 regularization in fully connected layer, used to mitigate
        overfitting to replay buffer during each episode of training

    Attributes (Needs Updating):
    -----------
    embedding : tf.keras.layers.Embedding
        embedding layer to map each cube piece id
    cnn : tf.keras.layers.Conv3d
        convolutional layer
    flatten : tf.keras.layers.Flatten
        flatten convolution output into one long vector
    fc : tf.keras.layers.Dense
        fully connected layer w/ output dimension equal to number of cube moves
    lr : tf.keras.layers.LeakyReLU
        activation layer for self.fc, chosen to avoid dying relu issue

    '''
    def __init__(self,
                 embed_dim=50,
                 num_filters=20,
                 num_conv_layers=1,
                 kernel_size=2,
                 regularization_constant=0,
                 num_dense_layers=1,
                 dense_activation='elu',
                 conv_activation='elu'):
        super(CNNWithBatchNorm, self).__init__()
        assert num_conv_layers > 0, "at least 1 conv layer required"
        assert num_dense_layers > 0, "at least 1 dense layer required"
        assert dense_activation=='elu'
        assert conv_activation=='elu'

        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.rc = regularization_constant
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.embedding = tf.keras.layers.Embedding(27, self.embed_dim)
        self.cnn = self._build_conv_layers(num_conv_layers,
                                           activation=self.conv_activation)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = self._build_fc_layers(num_dense_layers,
                                        activation=self.dense_activation)
        self.output_layer = tf.keras.layers.Dense(12,
                                            activation='elu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                            bias_regularizer=tf.keras.regularizers.l2(l=self.rc))

    def _build_conv_layers(self, num_conv_layers, activation='elu'):
        '''
        helper function for init
        '''
        cnn = tf.keras.Sequential()
        for i in range(num_conv_layers):
            if i == 0:
                padding = 'valid'
            else:
                padding = 'same'
            cnn_layer = tf.keras.layers.Conv3D(filters=self.num_filters,
                                               kernel_size=self.kernel_size,
                                               data_format='channels_last',
                                               padding = padding,
                                               activation=None)
            batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
            activation = tf.keras.layers.ELU()
            cnn.add(cnn_layer)
            cnn.add(batch_norm)
            cnn.add(activation)
        return cnn


    def _build_fc_layers(self, num_dense_layers, activation=None):
        '''
        helper function for init
        '''
        fc = tf.keras.Sequential()
        for i in range(num_dense_layers):
            fc_layer = tf.keras.layers.Dense(50,
                                             kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                             bias_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                             activation=None)
            batch_norm = tf.keras.layers.BatchNormalization(axis=1)
            activation = tf.keras.layers.ELU()

            fc.add(fc_layer)
            fc.add(batch_norm)
            fc.add(activation)
        return fc


    def call(self, x, training=False):
        '''
        Forward pass for CNN

        Parameters:
        -----------
        x : tensorflow.python.framework.ops.EagerTensor
            shape is (batch_size, 3, 3, 3) representing rubix cube
        '''
        x = self.embedding(x)
        x = self.cnn(x, training)
        x = self.flatten(x)
        x = self.fc(x, training)
        x = self.output_layer(x)
        return x

class WideNet(tf.keras.Model):
    '''
    Parameters:
    -----------
    regularization_constant : float
        lambda for l2 regularization in fully connected layer, used to mitigate
        overfitting to replay buffer during each episode of training

    Attributes (Needs Updating):
    -----------
    embedding : tf.keras.layers.Embedding
        embedding layer to map each cube piece id
    cnn : tf.keras.layers.Conv3d
        convolutional layer
    flatten : tf.keras.layers.Flatten
        flatten convolution output into one long vector
    fc : tf.keras.layers.Dense
        fully connected layer w/ output dimension equal to number of cube moves
    lr : tf.keras.layers.LeakyReLU
        activation layer for self.fc, chosen to avoid dying relu issue

    '''
    def __init__(self,
                 regularization_constant=0):
        super(WideNet, self).__init__()

        self.rc = regularization_constant
        self.embedding = tf.keras.layers.Embedding(27, 256)
        self.cnn = self._build_conv_layer()
        self.flatten = tf.keras.layers.Flatten()
        self.fc = self._build_fc_layers()
        self.output_layer = tf.keras.layers.Dense(12,
                                            activation='elu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                            bias_regularizer=tf.keras.regularizers.l2(l=self.rc))

    def _build_conv_layer(self):
        '''
        helper function for init
        '''
        cnn = tf.keras.Sequential()
        cnn_layer = tf.keras.layers.Conv3D(filters=128,
                                           kernel_size=3,
                                           data_format='channels_last',
                                           padding = 'same',
                                           activation=None)
        activation = tf.keras.layers.ELU()
        cnn.add(cnn_layer)
        cnn.add(activation)
        return cnn


    def _build_fc_layers(self):
        '''
        helper function for init
        '''
        fc = tf.keras.Sequential()

        fc_1 = tf.keras.layers.Dense(1024,
                                     kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                     bias_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                     activation='elu')
        fc_2 = tf.keras.layers.Dense(512,
                                     kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                     bias_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                     activation='elu')
        fc.add(fc_1)
        fc.add(fc_2)
        fc.add(fc_3)
        return fc


    def call(self, x, training=False):
        '''
        Forward pass for CNN

        Parameters:
        -----------
        x : tensorflow.python.framework.ops.EagerTensor
            shape is (batch_size, 3, 3, 3) representing rubix cube
        '''
        x = self.embedding(x)
        x = self.cnn(x, training)
        x = self.flatten(x)
        x = self.fc(x, training)
        x = self.output_layer(x)
        return x
