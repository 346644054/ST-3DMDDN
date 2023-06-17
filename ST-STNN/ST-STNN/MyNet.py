# -*- coding:utf-8 -*-
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.engine.topology import Layer

import numpy as np


class mDense2D():
    def __init__(self, input, nb_dense_block, nb_layers, growth_rate, nb_filter, base_filter=3,dropout_rate=None, weight_decay=1E-4):

        self.layer = self.DenseUnits2D(input, nb_dense_block, nb_layers, growth_rate, nb_filter, base_filter,dropout_rate,weight_decay)


    def _conv_block_2D(self,input, nb_filter, base_filter=3, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout
        Args:
            input: Input keras tensor
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added
        '''

        x = Activation('relu')(input)
        x = Convolution2D(nb_filter, (base_filter, base_filter), kernel_initializer="he_uniform", padding="same",
                          use_bias=False,dilation_rate=(2,2),
                          kernel_regularizer=l2(weight_decay))(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)

        return x

    def _dense_block_2D(self,x, nb_layers, nb_filter, growth_rate, base_filter=3, dropout_rate=None, weight_decay=1E-4):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        Args:
            x: keras tensor
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with nb_layers of conv_block appended
        '''

        # concat_axis = 1 if K.image_dim_ordering() == "th" else -1
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        feature_list = [x]

        for i in range(nb_layers):
            x = self._conv_block_2D(x, growth_rate, base_filter, dropout_rate, weight_decay)

            feature_list.append(x)
            x = Concatenate(axis=concat_axis)(feature_list)
            nb_filter += growth_rate

        return x, nb_filter

    def _transition_block_2D(self,input, nb_filter, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D
        Args:
            input: keras tensor
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
        '''

        concat_axis = 1 if K.image_dim_ordering() == "th" else -1

        x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                          kernel_regularizer=l2(weight_decay))(input)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        # x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)

        return x

    def DenseUnits2D(self,input, nb_dense_block=3, nb_layers=8, growth_rate=4, nb_filter=16, base_filter=3,dropout_rate=None,
                   weight_decay=1E-4):
        ''' Build the create_dense_net model
        Args:
            nb_classes: number of classes
            img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay
        Returns: keras tensor with nb_layers of conv_block appended
        '''

        concat_axis = 1

        # Initial convolution
        x = Convolution2D(nb_filter, (base_filter, base_filter), kernel_initializer="he_uniform", padding="same", use_bias=False,
                          kernel_regularizer=l2(weight_decay))(input)

        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = self._dense_block_2D(x, nb_layers, nb_filter, growth_rate, base_filter=base_filter,dropout_rate=dropout_rate,
                                       weight_decay=weight_decay)
            # add transition_block
            x = self._transition_block_2D(x, nb_filter,dropout_rate=dropout_rate, weight_decay=weight_decay)

        # The last dense_block does not have a transition_block
        x, nb_filter = self._dense_block_2D(x, nb_layers, nb_filter, growth_rate,base_filter=base_filter, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        return x

class myLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(myLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape