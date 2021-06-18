# from Shapedetector import ShapeDetector
from pyimagesearch.shapedetector import ShapeDetector
import imutils
import cv2

path = 'papersensor.jpg'
image = cv2.imread(path)
# resized = imutils.resize(image,width = 300)
# ratio = image.shape[0] / float(resized.shape[0])
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.threshold(blurred,200,225, cv2.THRESH_BINARY)[1]
mask = cv2.bitwise_and(resized,resized, mask=thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
idx =0
for c in cnts:
    idx+= 1
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
    # set values as what you need in the situation
        cX, cY = 0, 0
    shape = sd.detect(c)


    # c = c.astype("float")
    # c *= ratio
    # c = c.astype("int")

    # cv2.drawContours(mask, [c], -1, (0, 255, 0), 2)
    # cv2.rectangle(mask,)
    x,y,w,h = cv2.boundingRect(c)
    roi=mask[y:y+h,x:x+w]
    cv2.rectangle(mask,(x-10,y-10),(x+w+10,y+h+10),(200,0,0),2)
    # cv2.putText(mask, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        # 0.5, (255, 255, 255), 2)
    # show the output image
    cv2.imwrite(str(idx) + '.jpg', roi)
    cv2.imshow("Image", mask)
    cv2.waitKey(0)

