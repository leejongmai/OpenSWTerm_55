import cv2

dog_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("./image/hillo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



dogs = dog_cascade.detectMultiScale(gray, 1.01, 6, minSize = (135,135))
faces = face_cascade.detectMultiScale(gray, 1.4, 5, minSize = (120,130))

for(x,y,w,h) in dogs:
    print(x,y,w,h)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

for (x, y, w, h) in faces:
    print(x, y, w, h)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


print("number of dogs in image =", len(dogs))
print("number of people in image =", len(faces))

cv2.imshow("img", img)
cv2.waitKey()