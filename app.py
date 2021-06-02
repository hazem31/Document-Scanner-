from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
from multiprocessing import Process,freeze_support,Queue
import queue

from scipy.spatial import distance as dist
import numpy as np
import cv2

from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.artist import Artist

import matplotlib.pyplot as plt
import itertools
import math
from pylsd.lsd import lsd
import argparse
import os
import imutils


class PolygonInteractor(object):
    """
    An polygon editor
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def get_poly_points(self):
        return np.asarray(self.poly.xy)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)



def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    return np.array([tl, tr, br, bl], dtype = "float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def translate(image, x, y):
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	(h, w) = image.shape[:2]

	if center is None:
		center = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated
# inter area is important 
def resize(image, height , inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]


	r = height / float(h)
	dim = (int(w * r), height)

	
	resized = cv2.resize(image, dim, interpolation = inter)

	return resized


def valid(cnt, IM_WIDTH , IM_HEIGHT , area=0.2):
    return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * area)


def interactive_get_contour(screenCnt, rescaled_image):
    poly = Polygon(screenCnt, animated=True, fill=False, color="blue", linewidth=5)
    fig, ax = plt.subplots()
    ax.add_patch(poly)
    ax.set_title(('Close the window when finish adjusting the corners'))
    p = PolygonInteractor(ax, poly)
    plt.imshow(rescaled_image)
    plt.show()

    new_points = p.get_poly_points()[:4]
    new_points = np.array([[p] for p in new_points], dtype = "int32")
    return new_points.reshape(4, 2)


def get_corners(img):
        lines = lsd(img)
        corners = []
        if lines is not None:
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        return corners


def doc(location,mode=False,out_name='result' , area = 0.2):
    RESCALED_HEIGHT = 500.0
    image = cv2.imread(location)    
    assert(image is not None)

    ratio = image.shape[0] / RESCALED_HEIGHT
    orig = image.copy()
    rescaled_image = resize(image, height = int(RESCALED_HEIGHT))

    MORPH = 9
    High = 50
    Low = 0

    IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

    gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    edged = cv2.Canny(dilated, Low, High)


    # cnts , hier = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # screenCnt = None
    # flag = 0
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    #     if valid(approx,IM_WIDTH , IM_HEIGHT , area):
    #         screenCnt = approx
    #         screenCnt = screenCnt.reshape(4, 2)
    #         flag = 1
    #         break


    # if flag==0:
    #     TOP_RIGHT = (IM_WIDTH, 0)
    #     BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
    #     BOTTOM_LEFT = (0, IM_HEIGHT)
    #     TOP_LEFT = (0, 0)
    #     screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
    #     screenCnt = screenCnt.reshape(4, 2)

    test_corners = get_corners(edged)

    approx_contours = []

    if len(test_corners) >= 4:
        quads = []

        for quad in itertools.combinations(test_corners, 4):
            points = np.array(quad)
            points = order_points(points)
            points = np.array([[p] for p in points], dtype = "int32")
            quads.append(points)

        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
        #quads = sorted(quads, key=self.angle_range)

        approx = quads[0]
        if valid(approx, IM_WIDTH, IM_HEIGHT,area):
            approx_contours.append(approx)


    (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        approx = cv2.approxPolyDP(c, 80, True)
        if valid(approx, IM_WIDTH, IM_HEIGHT,area):
            approx_contours.append(approx)
            break

    if not approx_contours:
        TOP_RIGHT = (IM_WIDTH, 0)
        BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
        BOTTOM_LEFT = (0, IM_HEIGHT)
        TOP_LEFT = (0, 0)
        screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

    else:
        screenCnt = max(approx_contours, key=cv2.contourArea)

    screenCnt = screenCnt.reshape(4, 2)
    
    if mode:
        screenCnt = interactive_get_contour(screenCnt, rescaled_image)

    warped = four_point_transform(orig, screenCnt * ratio)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    sharpen = cv2.GaussianBlur(gray, (0,0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)



    if("out" not in os.listdir()):
        os.mkdir("out")
  
    directory,inputImageName = os.path.split(location)

    outpath=os.path.join(directory,"out")
 
    outimgpath=os.path.join(outpath,inputImageName)

    cv2.imwrite(outimgpath,thresh)
    
    

import cv2
import numpy as np

imagePath = 'wm2.jpg' 

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


def removeTransparentWatermark(filename, outputfilename):
    img = cv2.imread(filename)

    alpha = 2.0
    beta = -160

    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)

    cv2.imwrite(outputfilename, new)



######################################
#################GUI##################
######################################
   
def main():
    global GUI 

    GUI = Tk()
    GUI.title("Ultimate Image Processor")
    GUI.configure(bg='#d2d2d2')
    GUI.geometry("1000x700")
    GUI.resizable(True, True)
    
    labelbanner = Label(GUI, text="Ultimate Image Processor", font=("Arial", 28), bg='lightgreen', relief="ridge", fg="White")
    labelbanner.grid(columnspan=5, padx=250, sticky='ew')
    
    DrawScreen()
    
    GUI.mainloop()

    

def DrawScreen():
    DestroyMain()
    global ButtonSelectQueryImg,ButtonIndexDB
    global labelalg,LabelIndexNote
    global buttonHistoSim,buttonGlobalColor,buttonColorLayout
    global ButtonDocExtract,ButtonRemoveLogo,ButtonWatermark,ButtonTransparentWatermark
    global labelthres,labelslidernote
    global origx,origy
    global QueryImgPath
    global LabelSaveNote

    origx=0
    origy=-70
                
    ButtonSelectQueryImg = Button(GUI, text="Select Query Image", font=("Arial", 10), command=lambda: SelectQueryImg())
    ButtonSelectQueryImg.configure(height=2, width=14)
    ButtonSelectQueryImg.place(x=origx+70, y=origy+120)
    
    QueryImgPath = Text(GUI, height=2, width=40)


    ButtonDocExtract = Button(GUI, text="Extract Document",bg="lightgreen", font=("Arial", 9), command=lambda: ExtractDocument())
    ButtonDocExtract.configure(height=2, width=17)

    
    ButtonRemoveLogo = Button(GUI, text="Remove Logo",bg="lightgreen", font=("Arial", 9), command=lambda: removeLogoWrapper())
    ButtonRemoveLogo.configure(height=2, width=17)

    ButtonWatermark = Button(GUI, text="Remove Watermark",bg="lightgreen", font=("Arial", 9), command=lambda: removeWatermarkWrapper())
    ButtonWatermark.configure(height=2, width=17)
    
    ButtonTransparentWatermark = Button(GUI, text="Remove\n  TranspaentWatermark  ",bg="lightgreen", font=("Arial", 9), command=lambda: removeTransparentWatermarkWrapper())
    ButtonTransparentWatermark.configure(height=2, width=17)
    

    global SaveNote
    SaveNote=StringVar()
    
    
    
#########################################################

    
def SelectQueryImg(): 
    error = 0
    global queryimgpath
    queryimgpath = filedialog.askopenfilenames()
    try:
      queryimgpath=queryimgpath[0]
    except:
        error=1
        
    global labelqueryimg
    try:
        labelqueryimg.destroy()
    except:
               pass
    if (error == 0  and len(queryimgpath)!=0 ): #Display Query Image in UI
        im = Image.open(queryimgpath).resize((300, 400))
        ph = ImageTk.PhotoImage(im)
        
        labelqueryimg = Label(GUI,image=ph)
        labelqueryimg.image = ph
        labelqueryimg.place(x=0, y=origy+230)
        
        ButtonDocExtract.place(x=origx+10, y=origy+650)
        ButtonRemoveLogo.place(x=origx+150, y=origy+650)
        ButtonWatermark.place(x=origx+10, y=origy+700)
        ButtonTransparentWatermark.place(x=origx+150, y=origy+700)        
        
        QueryImgPath.delete('1.0', END)
        QueryImgPath.place(x=5, y=110)
        QueryImgPath.insert(END,queryimgpath )



def docWrappper(q,imglocation,mode=False,out_name="result",area=0.2):
   
    try:
       doc(imglocation,mode,out_name,area)
    except Exception as e:
       q.put(e)
    
    
def ExtractDocument():

     global labelresultimg
     
     try:
        labelresultimg.destroy()
     except:
         pass
        
     directory,inputImageName = os.path.split(queryimgpath)
     
     if("out" not in os.listdir()):
        os.mkdir("out")
        
     outpath=os.path.join(directory,"out")
     
     outimgpath=os.path.join(outpath,inputImageName)
     
     error=0
     try:       
        p1 = Process(target=docWrappper, args=(q,queryimgpath,True,inputImageName,0.2))
        p1.start()
        p1.join()
     except Exception as e:
         print(e)
         error=1
         try:
            print(q.get(0))
         except Exception as e2:
             print(e2)
             
     if(error==0):  
          updateOutputImgGUI(outimgpath)

def removeLogoWrapper():
    
     global labelresultimg
     
     try:
        labelresultimg.destroy()
     except:
         pass
        
     directory,inputImageName = os.path.split(queryimgpath)
     
     if("out" not in os.listdir()):
        os.mkdir("out")
        
     outpath=os.path.join(directory,"out")
     
     outimgpath=os.path.join(outpath,inputImageName)
     
     error=0
     
     removeLogo(queryimgpath, outimgpath)

     if(error==0):  
          updateOutputImgGUI(outimgpath)

    
def removeWatermarkWrapper():
    
     global labelresultimg
     
     try:
        labelresultimg.destroy()
     except:
         pass
        
     directory,inputImageName = os.path.split(queryimgpath)
     
     if("out" not in os.listdir()):
        os.mkdir("out")
        
     outpath=os.path.join(directory,"out")
     
     outimgpath=os.path.join(outpath,inputImageName)
     
     error=0
     
     removeWatermark(queryimgpath, outimgpath)

     if(error==0):  
          updateOutputImgGUI(outimgpath)
          
def removeTransparentWatermarkWrapper():
    
     global labelresultimg
     
     try:
        labelresultimg.destroy()
     except:
         pass
        
     directory,inputImageName = os.path.split(queryimgpath)
     
     if("out" not in os.listdir()):
        os.mkdir("out")
        
     outpath=os.path.join(directory,"out")
     
     outimgpath=os.path.join(outpath,inputImageName)
     
     error=0
     
     removeTransparentWatermark(queryimgpath, outimgpath)

     if(error==0):  
          updateOutputImgGUI(outimgpath)
          

#removeWatermark(imagePath, 'out.jpg')

#removeTransparentWatermark(imagePath, 'out2.png')
     
  
def updateOutputImgGUI(outImgPath):

     try:
         LabelSaveNote.destroy()
     except:
         pass
        
     LabelSaveNote= Label(GUI,textvariable=SaveNote,bg="#d2d2d2",fg="blue",font=("Times", 14))
     SaveNote.set("Image Saved At: \n"+outImgPath)
     LabelSaveNote.place(x=origx+390,y=origy+650)
     
     im = Image.open(outImgPath).resize((300, 400))
     ph = ImageTk.PhotoImage(im)

     labelresultimg=Label(GUI,image=ph)
     labelresultimg.image = ph
     labelresultimg.place(x=origx+450, y=origy+230)
     GUI.update()
                   



####################################
########_USEFUL_FUNCTIONS_##########
####################################
def ShowError(error,title="Error"):
    errorbox = Tk()
    errorbox.withdraw()
    messagebox.showinfo(title, error)
#####################################
def buttonselected(button, buttons):
    global selectedbutton
    selectedbutton = button
    for i in buttons:
        if (i == button):
            i.configure(bg="lightblue")
        else:
            i.configure(bg='SystemButtonFace')
#####################################
def space(word, numofspaces):
    space = ""
    for i in range(0, numofspaces - len(word)):
        space = space + " "
    return space
#####################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False  
########################################
def formatPath(path):
 return path.replace("\\","\\\\")
########################################
def openFile(path):
    path2=path.replace("\\","\\\\")
    
    path3=path2.replace(" ","^ ")
    
    os.system(path3)
########################################   
def readyExport():
    if("Exports" not in os.listdir()):
       os.mkdir("Exports")
    try:
     folders=os.listdir("Exports")
     exports=["export 0"]
     maxexport=0
     for folder in folders:
       if(folder.find("export")!=-1):
          exports.append(folder)
          num=int(folder.split()[1])
          if( num > maxexport ):
             maxexport=num
    except Exception as e:
       print(e)
       
    try:
        os.mkdir("Exports\\export "+str(maxexport+1))
        path="Exports\\export "+str(maxexport+1)        
    except Exception as e:
       print(e)
       
    return os.path.abspath(path)
########################################              
def exportFile(dstpath,srcpath):
    directory,vidName=os.path.split(srcpath)

    dstpath=os.path.join(dstpath,vidName)
   
    copyfile(srcpath,dstpath)
########################################    
def exportAll(paths,mode="oneFile"):
   dstpath=readyExport()
        
   paths2=[]
   
   if(isinstance(paths,str)):
     paths2.append(paths)
   else:
     paths2=paths
     
   for srcpath in paths2:
     print(srcpath)
     srcpath=srcpath.replace("Thumbnails\\","")
     srcpath=srcpath.replace(".png","")         

     exportFile(dstpath,srcpath)
     
   ShowError("Saved At: "+str(dstpath),"Done")

########################################     
def openVideo(thumbnailPath): #openVideoFromThumbNail full path
    path2=thumbnailPath.replace("\\","\\\\")   
    path3=path2.replace(" ","^ ")
    path4=path3.replace("Thumbnails\\\\","")
    path5=path4.replace(".png","")
   # print(path5)
    os.system(path5)
    
##########################
##########################
def DestroyMain():
  
    try:
       ButtonCBIR.destroy()
       ButtonCBVR.destroy()
    except:
        pass


    global labelthres,labelnote
    global LabelIndexNote

    
def DestroyCBIR():
    try:
       ButtonBack.destroy()
       ButtonSelectQueryImg.destroy()
       ButtonIndexDB.destroy()
       labelalg.destroy()
       LabelIndexNote.destroy()
       buttonHistoSim.destroy()
       buttonGlobalColor.destroy()
       buttonColorLayout.destroy()             
    except:
        pass
    try:
      QueryImgPath.destroy()
      ButtonFindMatches.destroy()
      labelqueryimg.destroy() 
    except:
        pass
    try:
        labelthres.destroy()
        labelslidernote.destroy()
        slider.destroy()
    except:
        pass
    DrawMainScreen()


def DestroyCBVR():
    try:
       ButtonBack.destroy()
       ButtonSelectQueryImg.destroy()
       ButtonIndexDB.destroy()
       labelalg.destroy()
       LabelIndexNote.destroy()
       buttonColorLayout.destroy()             
    except:
        pass
    try:
      QueryImgPath.destroy()
      LabelKfExtNote.destroy()
      ButtonExtractKF.destroy()
      ButtonFindMatches.destroy()
      labelqueryimg.destroy() 
    except:
        pass
    try:
        labelthres.destroy()
        labelslidernote.destroy()
        slider.destroy()
    except:
        pass
    DrawMainScreen()
    

class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom

#FindMatchesCBVR()
try:
    if __name__ == '__main__':
          q=Queue()
          freeze_support()
          main()
except:
   ShowError("Error Happened, Please Check Your Inputs!")
