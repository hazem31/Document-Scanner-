import cv2
import numpy as np

imagePath = '' 

def removeLogo(filename, outputfilename):
    # Read Image
    img = cv2.imread(filename)
    # Resize
    img = cv2.resize(img,(400,500))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Thresholding
    ret, gray = cv2.threshold(gray,127,255,0)
    gray2 = gray.copy()
    # Find Contours
    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv2.contourArea(cnt) < 5000:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(gray2,(x, y),(x + w, y + h), 0, -1)
    # Write Image
    cv2.imwrite(outputfilename,gray2)

removeLogo(imagePath, 'out.png')

def removeTransparentWatermark(filename, outputfilename):
    img = cv2.imread(filename)

    alpha = 2.0
    beta = -160

    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)

    cv2.imwrite(outputfilename, new)

removeTransparentWatermark(imagePath, 'out2.png')
