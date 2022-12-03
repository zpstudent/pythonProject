# -*- coding: utf-8 -*-
# @File:deep_learning_test3.py
# @Author:south wind
# @Date:2022-10-21
# @IDE:PyCharm
'''
* check callbacks such as tensorboard
* custom loss
* custom training

sources:
https://keras.io/guides/training_with_built_in_methods/
https://keras.io/guides/customizing_what_happens_in_fit/
'''

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error, mean_absolute_error
from keras.callbacks import TensorBoard
# %%
# prepare data
X = np.linspace(0, 0.8, 80)
X = np.reshape(X, [-1, 1])
test_func = lambda x: x ** 2 + x
Y = test_func(X)
plt.figure()
plt.plot(X, Y, '.')


# %%
# define a custom layer
# https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class PolyN(layers.Layer):
    def __init__(self, order, reg_strength, **kwargs):
        super(PolyN, self).__init__(**kwargs)
        self.order = order
        self.reg_strength = reg_strength

    def build(self, input_shape):
        self.coef = self.add_weight(
            shape=(self.order + 1,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        out = self.coef[self.order] * inputs + self.coef[self.order - 1]
        if self.order > 1:
            for i in range(self.order - 2, -1, -1):
                out = inputs * out + self.coef[i]

        self.add_loss(self.reg_strength * tf.reduce_sum(self.coef ** 2))  # add a layer related weights in "call" method
        self.add_metric(tf.reduce_sum(self.coef ** 2), name='coef_l2')
        return out


# %%

# simple regression network
x_in = layers.Input(shape=(1,))
x_out = PolyN(order=2, reg_strength=0.0, name='poly_coef')(x_in)

reg_M = Model(x_in, x_out)
reg_M.summary()

# %%
# define some callback functions
cb_tfb = TensorBoard(
    log_dir="./tf_board",
    update_freq="epoch",
)

# %%
# train the model
reg_M.compile(loss='mean_absolute_error',
              optimizer=Adam(learning_rate=1e-2))

history = reg_M.fit(X, Y, batch_size=40, epochs=80,
                    validation_split=0.2,
                    callbacks=[cb_tfb], shuffle=True)
plt.figure()
plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='val-loss')
plt.legend()

# %%
# prepare test data
X_test = np.linspace(0.8, 1.0, 10)[..., None]
Y_test = test_func(X_test)

# check prediction
Y_pred = reg_M.predict(X_test)
Y_train_pred = reg_M.predict(X)
plt.figure()
plt.plot(X, Y, 'k-')
plt.plot(X, Y_train_pred, 'r-')
plt.plot(X_test, Y_test, 'k--')
plt.plot(X_test, Y_pred, 'r--')
plt.axvline(0.8)
plt.xlim([0, 1])

# %%
reg_M.get_layer('poly_coef').weights
# %%
loss_tracker = tf.keras.metrics.Mean(name="loss")  # comment out related lines if you are using compiled loss
val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")  # comment out related lines if you are using compiled loss


class CustomModel(tf.keras.Model):
    @tf.function
    def train_step(self, data, **kwargs):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = mean_absolute_error(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update loss
        loss_tracker.update_state(loss)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        return {'loss': loss_tracker.result(), **{m.name: m.result() for m in self.metrics}}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        # self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        loss = mean_absolute_error(y, y_pred)
        val_loss_tracker.update_state(loss)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {'loss': val_loss_tracker.result(), **{m.name: m.result() for m in self.metrics}}


reg_M_overwritten = CustomModel(x_in, x_out)
reg_M_overwritten.summary()

reg_M_overwritten.compile(optimizer=Adam(learning_rate=1e-2))  # must speciy loss argument if using complied_loss above

history = reg_M_overwritten.fit(X, Y, batch_size=40, epochs=80,
                                validation_split=0.2,
                                shuffle=True)
plt.figure()
plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='val-loss')
plt.legend()

# %%
'''
check some gradient
'''
with tf.GradientTape() as tape:
    x_tensor = tf.constant(X[:2], dtype=tf.float32)
    tape.watch(x_tensor)
    y_pred = reg_M_overwritten(x_tensor, training=False)

# Compute gradients
Jac = tape.jacobian(y_pred,
                    x_tensor).numpy().squeeze()  # be aware: MUST include above lines if you need to run for the next time

# %%
