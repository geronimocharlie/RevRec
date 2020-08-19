# Deprecated, just quick sanity test and proof of concept in tensorflow due to the team being more experienced in tf. 

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from integration_task import Integration_Task
batch_size = 16
length = 100
steps = 100000
model = keras.Sequential()
print_every = 50
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
#model.add(layers.Dense(32, activation=tf.tanh))

# Add a LSTM layer with 128 internal units.
#model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.SimpleRNN(128, return_sequences=True))

# Add a Dense layer with 10 units.
model.add(layers.Dense(2, activation = tf.nn.softmax))

data_source = Integration_Task(length=length,batch_size=batch_size)
optimizer = keras.optimizers.Adam()
for i in range(steps):
    samples, targets = data_source.generate_sample()
    with tf.GradientTape() as tape:
        out = model(samples)
        loss = tf.reduce_mean(keras.losses.categorical_crossentropy(out, tf.squeeze(tf.one_hot(targets, depth=2))))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if i%print_every == 0:
        print(loss)
        out = np.argmax(out.numpy(), axis=-1)
        print(out.shape)
        targets = np.squeeze(targets)
        print(np.sum(out==targets)/(targets.shape[0]*targets.shape[1]))
