import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

model = models.load_model('save_at_25.keras')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((100,100))
        img_array = np.array(im)

        #Expand dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict function using keras
        prediction = model.predict(img_array)#[0][0]
        print(prediction)
        #Customize this part to your liking...
        if(prediction == 1 or prediction == 0):
            print("No Human")
        elif(prediction < 0.74 and prediction != 0):
            print("No glasses")
        elif(prediction > 0.74 and prediction != 1):
            print("Glasses")

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()