import cv2

face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

img = cv2.imread("resources/people.jpg")
result_img = img.copy()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow("Original image", img)
cv2.imshow("Result image", result_img)

cv2.waitKey(0)