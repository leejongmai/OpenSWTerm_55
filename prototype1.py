import cv2

cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

img = cv2.imread("./image/hello.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dogs = cat_cascade.detectMultiScale(gray, 1.05, 6, minSize = (20,20))

for(x,y,w,h) in dogs:
    print(x,y,w,h)
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)


print("number of dogs in image =", len(dogs))

cv2.imshow("img", img)
cv2.waitKey()