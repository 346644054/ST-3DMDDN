from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,Multiply,MaxPooling3D,GlobalMaxPooling3D,UpSampling3D,Add,Lambda,
    Embedding,
    Permute,Concatenate,Dropout,GlobalAveragePooling2D
)
import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import tensorflow as tf


K.set_image_data_format('channels_first')


def squeeze_excite_block2(input, filter, channel, map_height, map_width, data_format='channels_first', ratio=2,
                          layer_count=0):
    # filters=64
    # init = Reshape((filter*channel, map_height, map_width))(input)
    # x = MaxPooling3D(data_format=data_format)(input)
    x = GlobalMaxPooling3D(data_format="channels_first")(input)
    # x = K.max(input, axis=[2, 3, 4], keepdims=True)
    x = Reshape((filter, 1, 1, 1))(x)

    x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               data_format="channels_first")(x)
    x = Conv3D(filters=filter, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same',
               data_format="channels_first")(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)

    # x = Reshape((384, 1, 1))(x)
    # x = input * x  # bs, c, h, w
    x = Multiply()([input, x])
    x = Reshape((filter, channel, map_height, map_width))(x)  # bs,
    return x

def squeeze_excite_block1(input, filter, channel, map_height, map_width, data_format='channels_first', ratio=2, layer_count=0):
    init = Reshape((filter * channel, map_height, map_width))(input)
    # init = tf.reshape(input, (filter*channel, map_height, map_width))
    # channel_axis = 1 if data_format == "channels_first" else -1
    # filters = init.shape[channel_axis]
    h, w = init.shape[-2].value, init.shape[-1].value
    filters = h * w
    se_shape = (1, 1, 32, 32)

    # se = tf.reduce_mean(init, axis=1, keep_dims=True)  # bs, 1, h , w
    # se = tf.keras.backend.mean(init, axis=1, keepdims=True)  # bs, 1, h , w
    se = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(init)
    se = Reshape(se_shape)(se)
    x = Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               data_format="channels_first")(se)
    x = Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same',
               data_format="channels_first")(x)
    # x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    x = Activation("sigmoid")(x)

    # se = tf.reduce_mean(init, axis=1, keepdims=True)
    # se = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(init)
    # se = Reshape(se_shape)(se) # bs, 1, 1, h*w
    # # se = tf.reshape(se, se_shape)
    # se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    # se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = Reshape((1, 1, h, w))(x)  # bs, 1, h, w

    # if data_format == 'channels_first':
    #     se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])  # bs, c, h, w
    x = Reshape((filter, channel, map_height, map_width))(x)  # bs,
    return x




def dense_conv3D(nb_filter, len_closeness, nb_layers):
    def f2(input):
        # cnn_fea = []
        # tmp = input
        data_format = 'channels_first'
        cnn_out = [input]
        for i in range(nb_layers):
            if i == 0:
                tmp_in = cnn_out[0]
                cnn_out.append(tmp_in)
            else:
                if i < nb_layers:
                    tmp_in = Concatenate(axis=1)(cnn_out)
                    tmp = Conv3D(filters=nb_filter, kernel_size=(2, 3, 3), strides=(1, 1, 1),
                                 border_mode="same", dilation_rate=(1, i+1, i+1),
                                 kernel_initializer='random_uniform', data_format='channels_first')(tmp_in)
                    # conv = Conv3D(filters=64, kernel_size=(6, 3, 3), strides=(1, 1, 1), border_mode="same",
                    #               kernel_initializer='random_uniform')(input)

                    # tmp = squeeze_excite_block(input=tmp, len_seq=64, channel=6, map_height=32,
                    #                            map_width=32,
                    #                            data_format=data_format, ratio=2, layer_count=i)
                    # tmp = attention(input=tmp,data_format='channel_first',filters=32)

                    tmp = BatchNormalization(axis=1, scale=False)(tmp)
                    tmp = Activation("relu")(tmp)
                    cnn_out.append(tmp)
                else:
                    pass
                # tmp_in = Concatenate(axis=1)([cnn_out[i-1],cnn_out[i]])
                # tmp_in = Concatenate(axis=1,[cnn_out[0],cnn_out[1]])
                # tmp_in = Concatenate([cnn_out[0],cnn_out[1]],axis=1)
                # tmp_in = Concatenate(axis=1)(cnn_out)

            # tmp = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same",
            #              kernel_initializer='random_uniform',data_format='channels_first')(tmp_in)




            # tmp = squeeze_excite_block1(input=tmp, len_seq=3, channel=8, map_height=192,
            #                               map_width=192,
            #                               data_format=data_format, ratio=2, layer_count=i)



            # tmp = Conv3D(filters=nb_filter, kernel_size=kernel_size, strides=(1, 1, 1), padding='same',kernel_initializer='random_uniform',
            #                                 data_format='channels_first')(tmp_in)




        tmp1 = cnn_out[-1]
        tmp1 = Dropout(rate=0.5)(tmp1)
        # tmp2 = Reshape(([nb_filter * 16 * 8]))(tmp1)
        # tmp3 = Dense(units=dense_units, activation='tanh')(tmp2)
        # res = Dropout(rate=0.5)(tmp3)

        return tmp1

    return f2

def dense_conv3D1(nb_filter, len_closeness, nb_layers):
    def f2(input):
        # cnn_fea = []
        # tmp = input
        data_format = 'channels_first'
        cnn_out = [input]
        for i in range(nb_layers):
            if i == 0:
                tmp_in = cnn_out[0]
                cnn_out.append(tmp_in)
            else:
                if i < nb_layers-1:
                    tmp_in = Concatenate(axis=1)(cnn_out)
                    tmp = Conv3D(filters=nb_filter, kernel_size=(len_closeness, 3, 3), strides=(1, 1, 1),
                                 border_mode="same",
                                 kernel_initializer='random_uniform', data_format='channels_first')(tmp_in)
                    # conv = Conv3D(filters=64, kernel_size=(6, 3, 3), strides=(1, 1, 1), border_mode="same",
                    #               kernel_initializer='random_uniform')(input)

                    # tmp = squeeze_excite_block(input=tmp, len_seq=64, channel=6, map_height=32,
                    #                            map_width=32,
                    #                            data_format=data_format, ratio=2, layer_count=i)
                    # tmp = attention(input=tmp,data_format='channel_first',filters=32)

                    tmp = BatchNormalization(axis=1, scale=False)(tmp)
                    tmp = Activation("relu")(tmp)
                    cnn_out.append(tmp)
                else:
                    pass
                # tmp_in = Concatenate(axis=1)([cnn_out[i-1],cnn_out[i]])
                # tmp_in = Concatenate(axis=1,[cnn_out[0],cnn_out[1]])
                # tmp_in = Concatenate([cnn_out[0],cnn_out[1]],axis=1)
                # tmp_in = Concatenate(axis=1)(cnn_out)

            # tmp = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same",
            #              kernel_initializer='random_uniform',data_format='channels_first')(tmp_in)




            # tmp = squeeze_excite_block1(input=tmp, len_seq=3, channel=8, map_height=192,
            #                               map_width=192,
            #                               data_format=data_format, ratio=2, layer_count=i)



            # tmp = Conv3D(filters=nb_filter, kernel_size=kernel_size, strides=(1, 1, 1), padding='same',kernel_initializer='random_uniform',
            #                                 data_format='channels_first')(tmp_in)




        tmp1 = cnn_out[-1]
        # tmp1 = Dropout(rate=0.5)(tmp1)
        # tmp2 = Reshape(([nb_filter * 16 * 8]))(tmp1)
        # tmp3 = Dense(units=dense_units, activation='tanh')(tmp2)
        # res = Dropout(rate=0.5)(tmp3)

        return tmp1

    return f2


class iLayer(Layer):
    '''
    final weighted sum
    '''
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape

class Recalibration(Layer):
    '''
    channel-wise recalibration for closeness component
    '''
    def __init__(self, **kwargs):
        super(Recalibration, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: (batch, c, h, w)
        '''
        initial_weight_value = np.random.random((input_shape[1], 2, input_shape[2], input_shape[3])) # (c,2,h,w)
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

        super(Recalibration, self).build(input_shape)

    def call(self, x):
        '''
        x: (batch, c, h,w)
        '''
        double_x = tf.stack([x,x], axis=2)  # [(batch,c,h,w), (batch, c,h,w)] => (batch,c,2,h,w)
        return tf.reduce_sum(double_x*self.W, 1)  # (batch,2,h,w)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 2,input_shape[2],input_shape[3]  # (batch_size,2,h,w)


class Recalibration_T(Layer):
    '''
    channel-wise recalibration for weekly period component:
    '''
    def __init__(self,channel,**kwargs):
        super(Recalibration_T, self).__init__(**kwargs)
        self.channel = channel

    def build(self, input_shape):
        '''
        input_shape: (batch, c, h, w)
        '''
        initial_weight_value = np.random.random(input_shape[1]*2) # [2c,]:because output 2 channel
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

        super(Recalibration_T, self).build(input_shape)

    def call(self, x):
        '''
        x: (batch, c, h, w)
        '''
        nb_channel = self.channel
        _, _, map_height, map_width = x.shape
        W = tf.reshape(tf.tile(self.W, [map_height*map_width]),(nb_channel, 2, map_height, map_width))  # sharing channel-wsie weight on different positions in the weekly-period recalibration block
        double_x = tf.stack([x,x], axis=2) # stack [(batch, c,h, w)] = (batch, c, 2, h,w)
        return tf.reduce_sum(double_x*W, 1) # (batch, 2, h, w)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2,input_shape[2],input_shape[3]) # (batch_size,2,h,w)


def _shortcut(input, residual):
    return keras.layers.Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        '''
        input: (batch,c,h,w)
        '''
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
    return f


def _residual_unit(nb_filter):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter)(input)
        return input
    return f


def ST3DNet(c_conf=(6, 2, 16, 8),p_conf=(4,2,16,8), t_conf=(4, 2, 16, 8), external_dim=8, nb_residual_unit=4):
    len_closeness, nb_flow, map_height, map_width = c_conf
    # main input
    main_inputs = []
    outputs = []
    main_inputs1 = []
    if c_conf is not None:
        len_closeness, nb_flow, map_height, map_width = c_conf
        input3 = Input(shape=(nb_flow, len_closeness, map_height, map_width))  # (2,t_c,h,w)
        main_inputs.append(input3)
        main_inputs1.append(input3)

    # if t_conf is not None:
    #     trend, nb_flow, map_height, map_width = t_conf
    #     input1 = Input(shape=(nb_flow, trend, map_height, map_width))  # (2,t_c,h,w)
    #     main_inputs.append(input1)
    #     main_inputs1.append(input1)
    if p_conf is not None:
        len_seq, nb_flow, map_height, map_width = p_conf
        input2 = Input(shape=(nb_flow, len_seq, map_height, map_width))
        main_inputs.append(input2)
        main_inputs1.append(input2)
    if t_conf is not None:
        trend, nb_flow, map_height, map_width = t_conf
        input1 = Input(shape=(nb_flow, trend, map_height, map_width))  # (2,t_c,h,w)
        main_inputs.append(input1)
        main_inputs1.append(input1)
        input = Concatenate(axis=2)(main_inputs1)
        # Conv1 3D
        # conv = Conv3D(filters=64, kernel_size=(6, 3, 3), strides=(1, 1, 1), border_mode="same",
        #               kernel_initializer='random_uniform')(input)
        # conv = Activation("relu")(conv)
        #
        # # Conv2 3D
        # conv = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(conv)
        # conv = Activation("relu")(conv)
        #
        # # Conv3 3D
        # conv = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(conv)
        #
        # # (filter,1,height,width)
        # reshape = Reshape((64, map_height, map_width))(conv)
        #
        # # Residual 2D [nb_residual_unit] Residual Units
        # residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit)(reshape)
        #
        # output_c = Recalibration()(residual_output)
        len_closeness = 6
        nb_filter = 32
        nb_layers = 12
        # input = Permute((2, 1, 3, 4))(input)
        dense_output = dense_conv3D(nb_filter=nb_filter, len_closeness=6,
                                    nb_layers=nb_layers)(input)

        ###meiyou pengzhan
        channel = 6
        filter = 32
        # dense_output = squeeze_excite_block1(input=dense_output, filter=filter, channel=2, map_height=32,
        #                                      map_width=32,
        #                                      data_format='channel_first', ratio=2, layer_count=0)

        dense_output1 = squeeze_excite_block2(input=dense_output, filter=filter, channel=6, map_height=32,
                                              map_width=32,
                                              data_format='channel_first', ratio=2, layer_count=0)

        dense_output2 = squeeze_excite_block1(input=dense_output, filter=filter, channel=6, map_height=32,
                                              map_width=32,
                                              data_format='channel_first', ratio=2, layer_count=0)

        dense_output = keras.layers.Add()([dense_output1, dense_output2])

        dense_output = Reshape((filter * channel, map_height, map_width))(dense_output)

        dense_output = Convolution2D(nb_filter=8, nb_row=3,
                                     nb_col=3,
                                     border_mode="same")(dense_output)
        dense_output = Activation('relu')(dense_output)

        dense_output = Convolution2D(nb_filter=2, nb_row=3,
                                     nb_col=3,
                                     border_mode="same")(dense_output)
        dense_output = Activation('relu')(dense_output)



        outputs.append(dense_output)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # from .iLayer import iLayer
        # new_outputs = []
        # for output in outputs:
        #     new_outputs.append(iLayer()(output))
        # main_output = keras.layers.Add()(new_outputs)
        outputs = Concatenate(axis=1)(outputs)
        main_output = Recalibration()(outputs)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = keras.layers.Add()([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('relu')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model


# def ST3DNetE(c_conf=(6, 2, 16, 8), t_conf=(4, 2, 16, 8), external_dim=8, nb_residual_unit=4):
#     len_closeness, nb_flow, map_height, map_width = c_conf
#     nb = map_height * map_width
#     # main input
#     main_inputs = []
#     outputs = []
#     if len_closeness > 0:
#         input = Input(shape=(nb_flow, len_closeness, map_height, map_width))  # (2,t_c,h,w)
#         main_inputs.append(input)
#         # embedding
#         input = Input(shape=(nb))  # (2,t_c,h,w)
#         emb_input = np.arange(nb).reshape(-1, nb)  # (1, nb=hw)
#         emb_input = Embedding(nb, 64, input_length=nb)(emb_input)  # (1, nb=hw, 64)
#         emb_input = emb_input.reshape(1, map_height, map_width, 64)  # (1, h, w, 64)
#         emb_input = Permute((2, 3), input_shape=(map_height, map_width, 64))(emb_input)  # (1, h, 64, w)
#         emb_input = Permute((1, 2), input_shape=(map_height, map_width, 64))(emb_input)  # (1, 64, h, w)
#
#         # Conv1 3D
#         conv = Conv3D(filters=64, kernel_size=(6, 3, 3), strides=(1, 1, 1), border_mode="same",
#                       kernel_initializer='random_uniform')(input)
#
#         # add embed and con1 3D
#         conv = emb_input + conv
#
#         conv = Activation("relu")(conv)
#
#         # Conv2 3D
#         conv = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(conv)
#         conv = Activation("relu")(conv)
#
#         # Conv3 3D
#         conv = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(conv)
#
#         # (filter,1,height,width)
#         reshape = Reshape((64, map_height, map_width))(conv)
#
#         # Residual 2D [nb_residual_unit] Residual Units
#         residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit)(reshape)
#
#         output_c = Recalibration()(residual_output)
#         outputs.append(output_c)
#
#     if t_conf is not None:
#         len_seq, nb_flow, map_height, map_width = t_conf
#         input = Input(shape=(nb_flow, len_seq, map_height, map_width))
#         main_inputs.append(input)
#
#         conv = Conv3D(nb_filter=8, kernel_dim1=len_seq, kernel_dim2=1, kernel_dim3=1, border_mode="valid")(input)
#         conv = Activation('relu')(conv)
#
#         output_t = Reshape((8, map_height, map_width))(conv)
#         output_t = Recalibration_T(8)(output_t)
#
#         outputs.append(output_t)
#
#     # parameter-matrix-based fusion
#     if len(outputs) == 1:
#         main_output = outputs[0]
#     else:
#         # from .iLayer import iLayer
#         new_outputs = []
#         for output in outputs:
#             new_outputs.append(iLayer()(output))
#         main_output = keras.layers.Add()(new_outputs)
#
#     # fusing with external component
#     if external_dim != None and external_dim > 0:
#         # external input
#         external_input = Input(shape=(external_dim,))
#         main_inputs.append(external_input)
#         embedding = Dense(output_dim=10)(external_input)
#         embedding = Activation('relu')(embedding)
#         h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
#         activation = Activation('relu')(h1)
#         external_output = Reshape((nb_flow, map_height, map_width))(activation)
#         main_output = keras.layers.Add()([main_output, external_output])
#     else:
#         print('external_dim:', external_dim)
#
#     main_output = Activation('relu')(main_output)
#     model = Model(input=main_inputs, output=main_output)
#
#     return model
