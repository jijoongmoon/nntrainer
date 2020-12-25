import tensorflow as tf
import numpy as np
import imageio

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image = np.array(x_train[0],dtype=np.uint8)

imageio.imwrite('0.bmp', image.reshape(28,28))


