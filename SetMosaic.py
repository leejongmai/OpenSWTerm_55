import cv2

dog_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("OpenSW_term55/image/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



dogs = dog_cascade.detectMultiScale(gray, 1.01, 6, minSize = (135,135))
faces = face_cascade.detectMultiScale(gray, 1.4, 5, minSize = (120,130))

for(x,y,w,h) in dogs:
    print(x,y,w,h)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

for (x, y, w, h) in faces:
    print(x, y, w, h)
    cv2.rectangle(img, (x, y), (x + w - 20, y + h - 20), (255, 0, 0), 2)

    face_roi = img[y:y + h - 20, x:x+w-20]
    face_roi = cv2.GaussianBlur(face_roi, (99,99), 30)

    img[y:y + face_roi.shape[0], x:x + face_roi.shape[1]] = face_roi


print("number of dogs in image =", len(dogs))
print("number of people in image =", len(faces))

cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
