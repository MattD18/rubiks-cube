'''
module for small cnn based architecture
'''
import tensorflow as tf

class CNN(tf.keras.Model):
    '''
    Architecture which embeds each cube, then passes through 3x3 convolutions

    Attributes:
    -----------
    embed_dim : int
        dimension of embedding space in which each cube piece id is represented
    num_filter : int
        number of filters in convolutional layer
    kernel_size : int
        kernel size for convolutional layer (assumes square shape)
    fc_dim : int
        dimension of fully connected layers
    reg_constant : float
        lambda for l2 regularization in fully connected layer, used to mitigate
        overfitting to replay buffer during each episode of training
    conv_activation : str
    dense_activation : str
    embedding : tf.keras.layers.Embedding
    cnn_layers : tf.keras.Sequential
        convolutional layers
    flatten : tf.keras.layers.Flatten()
        flatten convolution output into one long vector
    fc_layers : tf.keras.layers.Dense
        fully connected layer w/ output dimension equal to number of cube moves
    output_layer : tf.keras.layers.Dense
    '''
    def __init__(self,
                 embed_dim=50,
                 num_filters=20,
                 num_conv_layers=1,
                 kernel_size=2,
                 fc_dim = 50,
                 regularization_constant=0.0,
                 num_dense_layers=1,
                 dense_activation='elu',
                 conv_activation='elu'):
        super(CNN, self).__init__()
        assert num_conv_layers > 0, "at least 1 conv layer required"
        assert num_dense_layers > 0, "at least 1 dense layer required"


        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fc_dim = fc_dim
        self.reg_constant = regularization_constant
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.embedding = tf.keras.layers.Embedding(27, self.embed_dim)
        self.cnn_layers = self._build_cnn_layers(num_conv_layers,
                                                activation=self.conv_activation)
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layers = self._build_fc_layers(num_dense_layers,
                                                activation=self.dense_activation)
        self.output_layer = tf.keras.layers.Dense(12,
                                            activation=self.dense_activation,
                                            kernel_regularizer=tf.keras.regularizers.l2(l=self.reg_constant),
                                            bias_regularizer=tf.keras.regularizers.l2(l=self.reg_constant))

    def _build_cnn_layers(self, num_conv_layers, activation=None):
        '''
        helper function for init, builds convolution part of model

        Parameters:
        -----------
        num_conv_layers : int
        activation : str

        Returns:
        --------
        cnn : tf.Keras.Sequential
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
        helper function for init, builds fully connected part of model

        Parameters:
        -----------
        num_dense_layers : int
        activation : str
        
        Returns:
        --------
        fc : tf.Keras.Sequential
        '''
        fc = tf.keras.Sequential()
        for i in range(num_dense_layers):
            fc_layer = tf.keras.layers.Dense(self.fc_dim,
                                             kernel_regularizer=tf.keras.regularizers.l2(l=self.reg_constant),
                                             bias_regularizer=tf.keras.regularizers.l2(l=self.reg_constant),
                                             activation=activation)
            fc.add(fc_layer)
        return fc


    def call(self, x, training=True):
        '''
        Forward pass for model

        Parameters:
        -----------
        x : tensorflow.python.framework.ops.EagerTensor
            shape is (batch_size, 3, 3, 3) representing rubix cube

        Returns:
        ---------
        out : tensorflow.python.framework.ops.EagerTensor
            shape is (batch_size, 12) representing Q-values over actions
        '''
        out = self.embedding(x)
        out = self.cnn_layers(out)
        out = self.flatten(out)
        out = self.fc_layers(out)
        out = self.output_layer(out)
        return out

