import cv2
import numpy as np
from PIL import Image
from keras import models

model = models.load_model('dataset/save_at_25.keras')
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resizing into dimensions you used while training
    im = im.resize((100, 100))
    img_array = np.array(im)

    # Expand dimensions to match the 4D Tensor shape.
    img_array = np.expand_dims(img_array, axis=0)

    # Calling the predict function using keras
    prediction = model.predict(img_array)  # [0][0]
    print(prediction)
    # Customize this part to your liking...
    if prediction == 1 or prediction == 0:
        cv2.putText(frame, 'NO PRESENCE DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

    elif prediction < 0.73 and prediction != 0:
        cv2.putText(frame, 'NO GLASSES : %s' % prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                    cv2.LINE_4)

    elif prediction > 0.73 and prediction != 1:
        cv2.putText(frame, 'GLASSES : %s ' % prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                    cv2.LINE_4)

    cv2.imshow("Prediction", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
