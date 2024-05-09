import cv2
import numpy as np

cap = cv2.VideoCapture('video/people_flow.mp4')

while True:
    _, frame = cap.read()

    frame = cv2.resize(frame, None,
                       fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    people_cascade_src = 'haarcascade_fullbody.xml'
    people_cascade = cv2.CascadeClassifier(people_cascade_src)
    people = people_cascade.detectMultiScale(closing, 1.1, 1)

    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the output
    cv2.imshow('People', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()

cv2.destroyAllWindows()
