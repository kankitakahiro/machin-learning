#%%
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import numpy as np

train, test = tf.keras.datasets.mnist.load_data()
train_data, train_labels = train
test_data, test_labels = test

train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_labels = np.array(train_labels, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

assert train_data.min() == 0.
assert train_data.max() == 1.
assert test_data.min() == 0.
assert test_data.max() == 1.
#%%
epochs = 10
batch_size = 250
learning_rate = 0.001
#%%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8,
                           strides=2,
                           padding='same',
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4,
                           strides=2,
                           padding='valid',
                           activation='relu'),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                           momentum=0.0, 
                           decay=0.0, 
                           nesterov=False)

loss = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.losses.Reduction.NONE)
#%%
model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=['accuracy'])
hist=model.fit(train_data, train_labels,
          epochs=epochs,
          validation_data=(test_data, test_labels),
          batch_size=batch_size)
#%%
# 正解率の評価
test_loss, test_acc = model.evaluate(test_data,  test_labels, batch_size = batch_size)

print('\nTest accuracy:', test_acc)
predictions = model.predict(test_data)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
