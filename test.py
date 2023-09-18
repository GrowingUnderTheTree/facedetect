import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('save_at_25.keras')

new_model.save('glassesornoglasses.h5')