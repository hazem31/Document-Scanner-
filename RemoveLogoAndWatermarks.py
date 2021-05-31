import cv2
import numpy as np

imagePath = '' 

def removeLogo(filename, outputfilename):
    img = cv2.imread(filename)
    img = cv2.resize(img,(400,500))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,127,255,0)
    gray2 = gray.copy()

    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv2.contourArea(cnt)<5000:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(gray2,(x,y),(x+w,y+h),0,-1)

    cv2.imwrite(outputfilename,gray2)

removeLogo(imagePath, 'out.png')

def removeWatermark(filename, outputfilename):
    # Load the image
    img = cv2.imread(filename)

    # Convert the image to grayscale
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make a copy of the grayscale image
    bg = gr.copy()

    # Apply morphological transformations
    for i in range(5):
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * i + 1, 2 * i + 1))
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel2)
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel2)

    # Subtract the grayscale image from its processed copy
    dif = cv2.subtract(bg, gr)

    # Apply thresholding
    bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dark = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Extract pixels in the dark region
    darkpix = gr[np.where(dark > 0)]

    # Threshold the dark region to get the darker pixels inside it
    darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Paste the extracted darker pixels in the watermark region
    bw[np.where(dark > 0)] = darkpix.T

    cv2.imwrite(outputfilename, bw)


removeWatermark(imagePath, 'out.jpg')

def removeTransparentWatermark(filename, outputfilename):
    img = cv2.imread(filename)

    alpha = 2.0
    beta = -160

    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)

    cv2.imwrite(outputfilename, new)

removeTransparentWatermark(imagePath, 'out2.png')