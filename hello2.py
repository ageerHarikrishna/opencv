import cv2
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture("elon_musk_royal_society.jpg")

ret, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detects faces of different sizes in the input image

eye_s = eye_cascade.detectMultiScale(gray,1.3, 6)
for (x,y,w,h) in eye_s:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
cv2.imshow('img',img)
k = cv2.waitKey(0)

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
