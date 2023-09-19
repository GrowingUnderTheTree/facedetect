import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

image_size = (100, 100)
batch_size = 128

model = tf.keras.models.load_model('dataset/save_at_25.keras')
img = keras.utils.load_img(
    "pictures/glasses/0060.png", target_size=image_size
)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% No Glasses and {100 * score:.2f}% Glasses.")
