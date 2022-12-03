# -*- coding: utf-8 -*-
# @File:deep_learning_test6.py
# @Author:south wind
# @Date:2022-11-16
# @IDE:PyCharm
# %%
'''
To Do:

    * Need a way for periodic expansion for the grid points transformed outside of [0,1]
      Done by using tf.math.mod()
'''

import tensorflow as tf
from tensorflow import keras
from keras import layers


class STN_1D(layers.Layer):
    '''
    1D spatial transformer,
    all channels share the same transform
    '''

    def __init__(self,
                 localization_net,
                 output_size,
                 fill_mode,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size  # output_size does not include batch
        self.fill_mode = fill_mode
        super(STN_1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        super(STN_1D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),  # time
                int(output_size[1]),  # channels
                )

    def call(self, X, mask=None):
        deformation = self.locnet.call(X)
        # Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(deformation, X, self.output_size)
        return output

    # copy along the second dimension, each row is a copy of an index
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size):

        batch_size = tf.shape(signal)[0]
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x, dtype='float32')
        scale = tf.cast(output_size[0] - 1, dtype='float32')
        # scale = tf.cast(output_size[0], dtype='float32')

        x = x * scale

        x0 = tf.cast(tf.floor(x), 'int32')  # left neibor index
        x1 = x0 + 1  # right neibor

        max_x = tf.cast(t_len - 1, dtype='int32')
        # max_x = tf.cast(t_len,  dtype='int32')
        # print('max_x:{}'.format(max_x))
        zero = tf.zeros([], dtype='int32')

        if self.fill_mode == 'constant':
            x0 = tf.clip_by_value(x0, zero, max_x)  # if constant padding
            x1 = tf.clip_by_value(x1, zero, max_x)
        elif self.fill_mode == 'period':
            x0 = tf.math.mod(x0, max_x + 1)  # if periodic padding
            x1 = tf.math.mod(x1, max_x + 1)
        else:
            raise "fill mode not implemented yet."

        pts_batch = tf.range(batch_size) * t_len
        flat_output_dimensions = output_size[0]
        base = repeat(pts_batch, flat_output_dimensions)

        ind_0 = base + x0
        ind_1 = base + x1

        #        flat_signal = tf.transpose(signal, (0,2,1))
        flat_signal = tf.reshape(signal, [-1, num_channels])
        flat_signal = tf.cast(flat_signal, dtype='float32')

        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        # print('left: {}'.format(x0[-2:]))
        # print('right: {}'.format(x1[-2:]))

        w_0 = tf.expand_dims(x1 - x, 1)
        w_1 = tf.expand_dims(x - x0, 1)

        # print('left weight: {}'.format(w_0[-2:]))
        # print('right weight: {}'.format(w_1[-2:]))

        w_0 = tf.clip_by_value(w_0, 0.0, 1.0)

        output = w_0 * pts_values_0 + (1 - w_0) * pts_values_1

        output = tf.reshape(output, (-1, output_size[0], output_size[1]))

        return output

    def _meshgrid(self, t_length):
        x_linspace = tf.linspace(0., 1.0, t_length)
        ones = tf.ones_like(x_linspace)
        indices_grid = tf.concat([x_linspace, ones], axis=0)
        #        return tf.reshape(indices_grid, [-1])
        return indices_grid

    def _transform(self, affine_transformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = output_size[0]
        #        num_channels = tf.shape(input_sig)[-1]

        indices_grid = self._meshgrid(t_len)

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 2, -1))

        # this line is necessary for tf.matmul to perform
        affine_transformation = tf.reshape(affine_transformation, (-1, 1, 2))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        #        print(indices_grid.shape)
        #        print(affine_transformation.shape)
        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        #        transformed_grid = indices_grid[:,0,:]

        x_s_flatten = tf.reshape(transformed_grid, [-1])
        print('transformed grid: {}'.format(x_s_flatten))

        transformed_vol = self._interpolate(input_sig,
                                            x_s_flatten,
                                            output_size)

        return transformed_vol


class STN_1D_multi_channel(layers.Layer):
    '''
    1D spatial transformer,
    Each channel has its own transformation
    '''

    def __init__(self,
                 localization_net,
                 output_size,
                 fill_mode='period',
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        self.fill_mode = fill_mode
        super(STN_1D_multi_channel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        #    self.trainable_weights = self.locnet.trainable_weights
        super(STN_1D_multi_channel, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),  # time
                int(output_size[1]),  # channels
                )

    def call(self, X, mask=None):
        # transformation, sig = X
        transformation = self.locnet.call(X)
        # Y = tf.expand_dims(X[...,0], 4) # only transform the first channel
        output = self._transform(transformation, X, self.output_size)
        return output

    # copy along the second dimension, each row is a copy of an index
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, signal, x, output_size):
        batch_size = tf.shape(signal)[0]
        #        print(tf.keras.backend.int_shape(signal))
        t_len = tf.shape(signal)[1]
        num_channels = tf.shape(signal)[-1]

        x = tf.cast(x, dtype='float32')
        scale = tf.cast(output_size[0] - 1, dtype='float32')

        x = x * scale

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1

        max_x = tf.cast(t_len - 1, dtype='int32')
        zero = tf.zeros([], dtype='int32')

        if self.fill_mode == 'constant':
            x0 = tf.clip_by_value(x0, zero, max_x)  # if constant padding
            x1 = tf.clip_by_value(x1, zero, max_x)
        elif self.fill_mode == 'period':
            x0 = tf.math.mod(x0, max_x + 1)  # if periodic padding
            x1 = tf.math.mod(x1, max_x + 1)
        else:
            raise "fill mode not implemented yet."

        pts_batch = tf.range(batch_size * num_channels) * t_len
        flat_output_dimensions = output_size[0]
        base = repeat(pts_batch, flat_output_dimensions)

        #        print(base.shape)
        #        print(x0.shape)
        ind_0 = base + x0
        ind_1 = base + x1

        flat_signal = tf.transpose(signal, (0, 2, 1))
        flat_signal = tf.reshape(flat_signal, [-1])

        #        flat_signal = tf.reshape(signal, [-1, num_channels] )
        flat_signal = tf.cast(flat_signal, dtype='float32')

        pts_values_0 = tf.gather(flat_signal, ind_0)
        pts_values_1 = tf.gather(flat_signal, ind_1)

        # x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')

        w_0 = x1 - x
        # w_1 = x - x0
        #        w_0 = tf.expand_dims(x1 - x, 1)
        #        w_1 = tf.expand_dims(x - x0, 1)

        w_0 = tf.clip_by_value(w_0, 0.0, 1.0)
        output = w_0 * pts_values_0 + (1 - w_0) * pts_values_1

        output = tf.reshape(output, (-1, output_size[1], output_size[0]))

        output = tf.transpose(output, (0, 2, 1))

        return output

    def _meshgrid(self, t_length):
        x_linspace = tf.linspace(0., 1.0, t_length)
        ones = tf.ones_like(x_linspace)
        indices_grid = tf.concat([x_linspace, ones], axis=0)
        #        return tf.reshape(indices_grid, [-1])
        return indices_grid

    def _transform(self, affine_transformation, input_sig, output_size):
        batch_size = tf.shape(input_sig)[0]
        t_len = output_size[0]
        num_channels = tf.shape(input_sig)[-1]

        indices_grid = self._meshgrid(t_len)

        indices_grid = tf.tile(indices_grid,
                               tf.stack([batch_size * num_channels]))
        indices_grid = tf.reshape(
            indices_grid, (batch_size, num_channels, 2, -1))
        #
        # this line is necessary for tf.matmul to perform
        affine_transformation = tf.reshape(
            affine_transformation, (-1, num_channels, 1, 2))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        #        print(indices_grid.shape)
        #        print(affine_transformation.shape)
        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        #        transformed_grid = indices_grid[:,0,:]

        x_s_flatten = tf.reshape(transformed_grid, [-1])

        transformed_vol = self._interpolate(input_sig,
                                            x_s_flatten,
                                            output_size)

        return transformed_vol

    # %%


'''
independent functions
'''


def repeat(x, num_repeats):  # copy along the second dimension, each row is a copy of an index
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])


def interpolate(signal, x, output_size, fill_mode='constant'):
    batch_size = tf.shape(signal)[0]
    t_len = tf.shape(signal)[1]
    num_channels = tf.shape(signal)[-1]

    x = tf.cast(x, dtype='float32')
    scale = tf.cast(output_size[0] - 1, dtype='float32')
    # scale = tf.cast(output_size[0], dtype='float32')

    x = x * scale

    x0 = tf.cast(tf.floor(x), 'int32')  # left neibor index
    x1 = x0 + 1  # right neibor

    max_x = tf.cast(t_len - 1, dtype='int32')
    # max_x = tf.cast(t_len,  dtype='int32')
    print('max_x:{}'.format(max_x))
    zero = tf.zeros([], dtype='int32')

    if fill_mode == 'constant':
        x0 = tf.clip_by_value(x0, zero, max_x)  # if constant padding
        x1 = tf.clip_by_value(x1, zero, max_x)
    elif fill_mode == 'period':
        x0 = tf.math.mod(x0, max_x + 1)  # if periodic padding
        x1 = tf.math.mod(x1, max_x + 1)
    else:
        raise "fill mode not implemented yet."

    pts_batch = tf.range(batch_size) * t_len
    flat_output_dimensions = output_size[0]
    base = repeat(pts_batch, flat_output_dimensions)

    ind_0 = base + x0
    ind_1 = base + x1

    #        flat_signal = tf.transpose(signal, (0,2,1))
    flat_signal = tf.reshape(signal, [-1, num_channels])
    flat_signal = tf.cast(flat_signal, dtype='float32')

    pts_values_0 = tf.gather(flat_signal, ind_0)
    pts_values_1 = tf.gather(flat_signal, ind_1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    print('left: {}'.format(x0[-2:]))
    print('right: {}'.format(x1[-2:]))

    w_0 = tf.expand_dims(x1 - x, 1)
    w_1 = tf.expand_dims(x - x0, 1)

    print('left weight: {}'.format(w_0[-2:]))
    print('right weight: {}'.format(w_1[-2:]))

    w_0 = tf.clip_by_value(w_0, 0.0, 1.0)

    output = w_0 * pts_values_0 + (1 - w_0) * pts_values_1

    output = tf.reshape(output, (-1, output_size[0], output_size[1]))

    return output


def meshgrid(t_length):
    x_linspace = tf.linspace(0., 1.0, t_length)
    ones = tf.ones_like(x_linspace)  # for the purpose of shift calculation
    indices_grid = tf.concat([x_linspace, ones], axis=0)
    #        return tf.reshape(indices_grid, [-1])
    return indices_grid


def transform(affine_transformation, input_sig, output_size, fill_mode):
    batch_size = tf.shape(input_sig)[0]
    t_len = output_size[0]
    #        num_channels = tf.shape(input_sig)[-1]

    indices_grid = meshgrid(t_len)

    indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
    indices_grid = tf.reshape(indices_grid, (batch_size, 2, -1))

    # this line is necessary for tf.matmul to perform
    affine_transformation = tf.reshape(affine_transformation, (-1, 1, 2))
    affine_transformation = tf.cast(affine_transformation, 'float32')

    #        print(indices_grid.shape)
    #        print(affine_transformation.shape)
    transformed_grid = tf.matmul(affine_transformation, indices_grid)
    #        transformed_grid = indices_grid[:,0,:]

    x_s_flatten = tf.reshape(transformed_grid, [-1])
    print('transformed grid: {}'.format(x_s_flatten))

    transformed_vol = interpolate(input_sig,
                                  x_s_flatten,
                                  output_size,
                                  fill_mode)

    return transformed_vol


# %%
'''
cases for multi channels
'''


def interpolate_Nchannel(signal, x, output_size, fill_mode='period'):
    batch_size = tf.shape(signal)[0]
    #        print(tf.keras.backend.int_shape(signal))
    t_len = tf.shape(signal)[1]
    num_channels = tf.shape(signal)[-1]

    x = tf.cast(x, dtype='float32')
    scale = tf.cast(output_size[0] - 1, dtype='float32')

    x = x * scale

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1

    max_x = tf.cast(t_len - 1, dtype='int32')
    zero = tf.zeros([], dtype='int32')

    if fill_mode == 'constant':
        x0 = tf.clip_by_value(x0, zero, max_x)  # if constant padding
        x1 = tf.clip_by_value(x1, zero, max_x)
    elif fill_mode == 'period':
        x0 = tf.math.mod(x0, max_x + 1)  # if periodic padding
        x1 = tf.math.mod(x1, max_x + 1)
    else:
        raise "fill mode not implemented yet."

    pts_batch = tf.range(batch_size * num_channels) * t_len
    flat_output_dimensions = output_size[0]
    base = repeat(pts_batch, flat_output_dimensions)

    #        print(base.shape)
    #        print(x0.shape)
    ind_0 = base + x0
    ind_1 = base + x1

    flat_signal = tf.transpose(signal, (0, 2, 1))
    flat_signal = tf.reshape(flat_signal, [-1])

    #        flat_signal = tf.reshape(signal, [-1, num_channels] )
    flat_signal = tf.cast(flat_signal, dtype='float32')

    pts_values_0 = tf.gather(flat_signal, ind_0)
    pts_values_1 = tf.gather(flat_signal, ind_1)

    # x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')

    w_0 = x1 - x
    # w_1 = x - x0
    #        w_0 = tf.expand_dims(x1 - x, 1)
    #        w_1 = tf.expand_dims(x - x0, 1)

    w_0 = tf.clip_by_value(w_0, 0.0, 1.0)
    output = w_0 * pts_values_0 + (1 - w_0) * pts_values_1

    output = tf.reshape(output, (-1, output_size[1], output_size[0]))

    output = tf.transpose(output, (0, 2, 1))

    return output


def transform_Nchannel(
        affine_transformation,
        input_sig,
        output_size,
        fill_mode):
    batch_size = tf.shape(input_sig)[0]
    t_len = output_size[0]
    num_channels = tf.shape(input_sig)[-1]

    indices_grid = meshgrid(t_len)

    indices_grid = tf.tile(indices_grid, tf.stack([batch_size * num_channels]))
    indices_grid = tf.reshape(indices_grid, (batch_size, num_channels, 2, -1))
    #
    # this line is necessary for tf.matmul to perform
    affine_transformation = tf.reshape(
        affine_transformation, (-1, num_channels, 1, 2))
    affine_transformation = tf.cast(affine_transformation, 'float32')

    #        print(indices_grid.shape)
    #        print(affine_transformation.shape)
    transformed_grid = tf.matmul(affine_transformation, indices_grid)
    #        transformed_grid = indices_grid[:,0,:]

    x_s_flatten = tf.reshape(transformed_grid, [-1])

    transformed_vol = interpolate_Nchannel(input_sig,
                                           x_s_flatten,
                                           output_size,
                                           fill_mode)

    return transformed_vol


# %%
if __name__ == '__main__':
    '''
    Test for STN_1d
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    num_T = 20
    num_C = 2

    gridX = np.linspace(0, 1, num_T)
    valX = np.sin(2 * np.pi * gridX) + 0.2
    valX2 = np.c_[valX[..., None], -valX[..., None]]

    #     loc_net = keras.Sequential(
    #     [
    #         keras.Input(shape=(20,1)),
    #         layers.Conv1D(8, kernel_size=3, activation="elu", name='Conv_1'),
    #         layers.Dense(2, activation="linear", name='Dense'), # in ax+b, a>0  # need constraints
    #     ]
    # )

    #     stn_1D = STN_1D(loc_net, (num_T, num_C))

    # %%
    input_X = tf.constant(valX2[None, ...])
    affine_W = tf.constant(np.array([[0.5, 0.5]]))

    y = transform(affine_transformation=affine_W,
                  input_sig=input_X, output_size=(num_T, num_C))

    plt.subplot(2, 2, 1)
    plt.plot(valX2[:, 0], '-^')
    plt.subplot(2, 2, 2)
    plt.plot(valX2[:, 1], '-^')
    plt.subplot(2, 2, 3)
    plt.plot(y[0, :, 0].numpy(), '-^')
    plt.subplot(2, 2, 4)
    plt.plot(y[0, :, 1].numpy(), '-^')

    # %%
    '''
    test for the multi channel case
    '''

    affine_W2 = tf.constant(np.array([[[0.5, 0.5], [2, 0]]]))

    yy = transform_Nchannel(affine_transformation=affine_W2,
                            input_sig=input_X, output_size=(num_T, 2))

    plt.subplot(2, 2, 1)
    plt.plot(valX2[:, 0], '-^')
    plt.subplot(2, 2, 2)
    plt.plot(valX2[:, 1], '-^')
    plt.subplot(2, 2, 3)
    plt.plot(yy[0, :, 0].numpy(), '-^')
    plt.subplot(2, 2, 4)
    plt.plot(yy[0, :, 1].numpy(), '-^')

    # %%
    '''
    Test in the senario of curve matching
    '''

    class Coef_process(layers.Layer):
        '''
        This zoom will interrupt frequency information
        anticipate to perform well with shape patterns
        '''

        def __init__(self, alpha=1.0, beta=0.5, **kwargs):
            super(Coef_process, self).__init__(**kwargs)
            self.alpha = alpha
            self.beta = beta

        def call(self, X):
            x_slope = X[..., 0]  # batch, n_channels, 1
            x_bias = X[..., 1]

            # Pi = tf.constant(np.pi, dtype=tf.float32)
            '''will be interesting to try different functions'''
            # x_slope = tf.math.exp(-self.alpha*(x_slope-1.0)**2) # never greater than 1 in this case, other possibles: 1+k*tanh(x)
            # x_slope = tf.clip_by_value(x_slope, 1-self.alpha, 1+self.alpha)
            # # alpha value better be smaller for this case, must < 1
            x_slope = 1 + self.alpha * tf.math.tanh(x_slope)

            x_bias = self.beta * tf.math.tanh(x_bias)
            # x_bias = tf.clip_by_value(x_bias, -self.beta, self.beta)

            return tf.stack([x_slope, x_bias], axis=-1)  # batch, n_channels, 2

    def LT_network(input_shape, alpha, beta):
        '''
        For provding the parameter tensor required by the transformation
        '''
        x_in = layers.Input(input_shape)
        x = layers.Conv1D(
            8,
            kernel_size=3,
            activation="elu",
            name='Conv_1',
            padding='same')(x_in)
        x = layers.Flatten()(x)
        # x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(2 * input_shape[-1],
                         activation="linear",
                         name='Dense')(x)  # batch 2*nChannels
        x = layers.Reshape((input_shape[-1], 2))(x)  # batch, nChannels, 2
        x_out = Coef_process(alpha, beta)(x)  # batch, nChannels, 2

        return keras.models.Model(x_in, x_out)

    # %%
    loc_net = LT_network((20, 2), alpha=0.8, beta=0.6)
    x_in = layers.Input((20, 2))
    x_out = STN_1D_multi_channel(loc_net, output_size=(
        20, 2), fill_mode='period', name='affine')(x_in)
    test_model = keras.models.Model(x_in, x_out)

    test_model.compile(
        loss='mean_absolute_error',
        optimizer=keras.optimizers.Adam(
            lr=1e-3))
    test_model.summary()
    loc_net.summary()

    Y = y.numpy()

    # %%
    hist = test_model.fit(valX2[None, ...], Y, epochs=100)
    plt.plot(hist.history['loss'])

    Ypred = test_model.predict(valX2[None, ...])
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(valX2[:, 0], '-^')
    plt.subplot(2, 2, 2)
    plt.plot(valX2[:, 1], '-^')
    plt.subplot(2, 2, 3)
    plt.plot(Ypred[0, :, 0], '-^')
    plt.subplot(2, 2, 4)
    plt.plot(Ypred[0, :, 1], '-^')

    # %%
    affine_params = test_model.layers[1].locnet.predict(valX2[None, ...])
    print(affine_params)

# %%
