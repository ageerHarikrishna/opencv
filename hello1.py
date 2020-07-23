import cv2, glob
detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
all_images=glob.glob("*.jpg")
for image in all_images:
    img=cv2.imread(image)
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detect.detectMultiScale(grey_img,1.1,3)
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        final_img=cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("image", final_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()




