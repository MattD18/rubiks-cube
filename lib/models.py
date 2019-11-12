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
                 dense_activation='elu'):
        super(CNN, self).__init__()
        assert num_conv_layers > 0, "at least 1 conv layer required"
        assert num_dense_layers > 0, "at least 1 dense layer required"


        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.rc = regularization_constant
        self.dense_activation = dense_activation
        self.embedding = tf.keras.layers.Embedding(27, self.embed_dim)
        self.cnn = self._build_conv_layers(num_conv_layers)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = self._build_fc_layers(num_dense_layers)
        self.output_layer = tf.keras.layers.Dense(12,
                                            activation=self.dense_activation,
                                            kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                            bias_regularizer=tf.keras.regularizers.l2(l=self.rc))

    def _build_conv_layers(self, num_conv_layers):
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
                                               padding = padding)
            cnn.add(cnn_layer)
        return cnn


    def _build_fc_layers(self, num_dense_layers):
        '''
        helper function for init
        '''
        fc = tf.keras.Sequential()
        for i in range(num_dense_layers):
            fc_layer = tf.keras.layers.Dense(50,
                                             kernel_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                             bias_regularizer=tf.keras.regularizers.l2(l=self.rc),
                                             activation=self.dense_activation)
            fc.add(fc_layer)
        return fc


    def call(self, x):
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
