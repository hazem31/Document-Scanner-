#!pip install pytesseract
#!sudo apt install tesseract-ocr
# Import required packages
import cv2
import matplotlib.pyplot as plt
import pytesseract
from google.colab.patches import cv2_imshow
import numpy as np
from sklearn.cluster import KMeans


# Mention the installed location of Tesseract-OCR in your system "drive/MyDrive/sample.jpg"
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
def text_extract(img_path):  
# Read image from which text needs to be extracted
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Preprocessing the image starts
  
# Convert the image to gray scale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Performing OTSU threshold
  ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  
# Specify structure shape and kernel size. 
# Kernel size increases or decreases the area 
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect 
# each word instead of a sentence.
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
  
# Appplying dilation on the threshold image
  dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
  
# Finding contours
  contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_NONE)
  
# Creating a copy of image
  im2 = img.copy()
  
# A text file is created and flushed
# file = open("drive/MyDrive/recognized.txt", "w+")
# file.write("")
# file.close()
  
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
  iti=0
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      iti=iti+1  
    # Drawing a rectangle on copied image
      rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 0), 2)
      cv2.putText(rect, str(iti), (x, y+100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
     # plt.imshow(rect)  
    # Cropping the text block for giving input to OCR
      cropped = im2[y:y + h, x:x + w]
      
    # Open the file in append mode
      file = open("drive/MyDrive/recognized.txt", "a")
      
    # Apply OCR on the cropped image
      text = pytesseract.image_to_string(cropped) 
    # Appending the text into file
      file.write(text)
      file.write("\n")
      
    # Close the file
      file.close
  return rect,file
img,text= text_extract("drive/MyDrive/sample.jpg")
plt.imshow(img)
print(text)





def max_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    max_percent=0
    max_rgb_values=[]
    for i in range( len(colors)):
      if(colors[i][0]>max_percent):
        max_percent=colors[i][0]
        max_rgb_values=colors[i][1]

   
    return max_rgb_values


def delete_text(image_path,rect_num):
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Preprocessing the image starts
  reshape = img.reshape((img.shape[0] * img.shape[1], 3))
  cluster = KMeans(n_clusters=5).fit(reshape)
  visualize = max_colors(cluster, cluster.cluster_centers_)  
# Convert the image to gray scale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Performing OTSU threshold
  ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  
# Specify structure shape and kernel size. 
# Kernel size increases or decreases the area 
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect 
# each word instead of a sentence.
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
# Appplying dilation on the threshold image
  dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
# Finding contours
  contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_NONE) 
# Creating a copy of image
  im2 = img.copy()
  iti=0
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      iti=iti+1  
      if(iti in rect_num):
        rect=cv2.rectangle(im2, (x, y), (x + w, y + h), (visualize), -1)
        #plt.imshow(rect)
  plt.imshow(im2)






delete_text('drive/MyDrive/sample.jpg',[2])