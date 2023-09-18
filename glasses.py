import cv2
import matplotlib.pyplot as plt
count = 0

cam = cv2.VideoCapture(0)
for i in range(1000):
    check, frame = cam.read()
    cv2.imshow('Videos', frame)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(grayscale, (100, 100))
    cv2.imshow('grayscale', resize)
    cv2.imwrite("pictures//glasses//%04d.png" % count, resize)
    count += 1

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()