import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cnt = 0

while True:
    ret, frame = cap.read()
    cv2.imshow('d', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('c'):
        cv2.imwrite(f'./data/images/captured_{cnt}.png', frame)
        cnt += 1

cap.release()
cv2.destroyAllWindows()

