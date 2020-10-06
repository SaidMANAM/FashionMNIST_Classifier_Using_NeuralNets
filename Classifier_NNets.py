import tensorflow as tf
from tensorflow import keras

data = keras.datasets.fashion_mnist
print(data)
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
X_train = x_train[5000:]
Y_train = y_train[5000:]
X_val = x_train[:5000]
Y_val = y_train[:5000]
model = tf.keras.Sequential(
    [tf.keras.layers.Flatten(input_shape=[28, 28]), tf.keras.layers.Dense(300, activation='relu'),
     tf.keras.layers.Dense(100, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=70)
model.evaluate(x_test, y_test)
# import pandas as pd
# import matplotlib.pyplot as plt

# pd.DataFrame(history.history).plot()
# plt.show()
