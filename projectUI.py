import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as plt

plt.use("Qt5Agg")  # 声明使用QT5

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication ,Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, qRgb, QFont, qRed, qGreen, qBlue, QColor
from matplotlib import pyplot as plt

import cv2
import numpy as np

import argparse
from yolo import YOLO, detect_video
from PIL import Image

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        #初始化視窗大小
        self.title = "Yolo demo"
        self.left = 100
        self.top = 20
        self.width = 1200
        self.height = 1000
        #初始化按鈕與Label
        self.initUI()
        self.center()#置中
        self.show()

    def center(self):  #將畫面移至中間
        screen = QDesktopWidget().screenGeometry()  
        size = self.geometry()        
        self.move((screen.width() - size.width()) / 2,  (screen.height() - size.height()) / 2) 
        
    def initUI(self):
        #設定視窗
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.barLength = 255
        
        #Load
        #新建按鈕
        button_load = QPushButton("&Load", self)
        #按鈕的提示訊息
        button_load.setToolTip("Load the image")
        #按鈕座標
        button_load.move(1100, 10)
        #按鈕觸發事件
        button_load.clicked.connect(self.loadImg)
        
        #slider的Label
        self.l1 = QLabel('Canny threshold min 1 = 100',self)        
        self.l1.setGeometry(40, 680, 200,50)
        
        #Canny threshold slider
        self.sld1 = QSlider(Qt.Horizontal,self)
        self.sld1.setGeometry(40,730,200,50)
        self.sld1.setMinimum(0)
        self.sld1.setMaximum(500)
        self.sld1.setTickPosition(QSlider.TicksRight)
        self.sld1.setTickInterval(10)
        self.sld1.setValue(100)
        self.sld1.valueChanged[int].connect(self.displayEachObject)
        
        #slider的Label
        self.l2 = QLabel('Canny threshold min 2 = 200',self)        
        self.l2.setGeometry(40,780,200,50)
        
        #Canny threshold slider
        self.sld2 = QSlider(Qt.Horizontal,self)
        self.sld2.setGeometry(40,830,200,50)
        self.sld2.setMinimum(0)
        self.sld2.setMaximum(500)
        self.sld2.setTickPosition(QSlider.TicksRight)
        self.sld2.setTickInterval(10)
        self.sld2.setValue(200)
        self.sld2.valueChanged[int].connect(self.displayEachObject)
        
        #slider的Label
        self.l3 = QLabel('draw size = 200',self)        
        self.l3.setGeometry(340,780,200,50)
        
        #Canny threshold slider
        self.sld3 = QSlider(Qt.Horizontal,self)
        self.sld3.setGeometry(340,830,200,50)
        self.sld3.setMinimum(0)
        self.sld3.setMaximum(1000)
        self.sld3.setTickPosition(QSlider.TicksRight)
        self.sld3.setTickInterval(10)
        self.sld3.setValue(200)
        self.sld3.valueChanged[int].connect(self.displayEachObject)       
 
        
        
    def loadImg(self):
        #choose the file path
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                    "選取文件",
                                    "./",
                                    "All Files (*)")
        
        self.Image = Image.open(fileName1)
        self.ImgForCut = self.Image.copy()
        
#         cv2.cvtColor(self.cvImage, cv2.COLOR_BGR2RGB, self.cvImage)
        self.YOLO_call()
        
    #中文路徑解決方法
    def cv_imread(self,filePath):
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        return cv_img
    
    def YOLO_call(self):     #call the YOLO algo. 
#         pilImage = Image.fromarray(self.cvImage)
        y = YOLO()
        self.pilImage, self.label_dict = y.detect_image(self.Image)
#         self.cvImage = cv2.cvtColor(np.array(self.pilImage), cv2.COLOR_RGB2BGR)
        self.cvImage = np.array(self.pilImage)
        height, width, byteValue = self.cvImage.shape
        byteValue = byteValue * width
        self.mQImage = QImage(self.cvImage, width, height, byteValue, QImage.Format_RGB888)
        self.paint()
        print(self.label_dict)
        self.displayEachObject()
        
    def displayEachObject(self): #display each object in scroll area
        self.l1.setText('Canny threshold min 1 = '+str(self.sld1.value()))
        self.l2.setText('Canny threshold min 2 = '  +str(self.sld2.value()))
        self.l3.setText('draw size = '+str(self.sld3.value()))

        
        img = np.array(self.ImgForCut)
        object_list = list(self.label_dict.keys())
        self.Object_scrollArea = QScrollArea(self, frameWidth=0, frameShape=QScrollArea.NoFrame)
        self.Object_scrollArea.setGeometry(850,50,400,900)
        self.Object_scrollArea.setWidgetResizable(True)
        
        #scroll area layout
        self.grid = QGridLayout(self)
        
        self.grid.setSpacing(10)  #region width
        
        for i,obj in enumerate(object_list):

            prob, pt1, pt2 = self.label_dict[obj]
            cutImg = self.cut(img, pt1, pt2)
            cutImg_cv2 = cv2.cvtColor(np.array(cutImg), cv2.COLOR_RGB2BGR)
            self.displayEachObject_EachRow(i, obj, cutImg, prob, pt1, pt2)

        #add to scroll area
        self.grid.setAlignment(Qt.AlignCenter)        
        self.Object_scrollArea.setLayout(self.grid)
        self.Object_scrollArea.show()
        
    def displayEachObject_EachRow(self, i, name, cutImg,prob, pt1, pt2): 
        
        #img column
        label_img = QLabel()
        
        self.cvImage = np.array(cutImg)
        height, width, byteValue = self.cvImage.shape
        byteValue = byteValue * width
        mCutImage = QImage(self.cvImage, width, height, byteValue, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(mCutImage)
        pixmap_cutImg = QPixmap(pixmap)
        label_img.setPixmap(pixmap_cutImg)
        label_img.setAlignment(Qt.AlignCenter)
        label_img.setScaledContents(True) # keep the origin image size
        self.grid.addWidget(label_img, 3*i, 0, 3, 1)
        
        #item name column
        title_name = QLabel('Name')
        title_prob = QLabel('Probability')
        title_region = QLabel('Region')

        self.grid.addWidget(title_name,3*i,1)
        self.grid.addWidget(title_prob,3*i+1,1)
        self.grid.addWidget(title_region,3*i+2,1)
        
        #value column
        value_name = QLabel(name)
        value_prob = QLabel(str(prob))
        value_region = QLabel(str(pt1)+'  '+str(pt2))
        
        self.grid.addWidget(value_name,3*i,2)
        self.grid.addWidget(value_prob,3*i+1,2)
        self.grid.addWidget(value_region,3*i+2,2)
        
    def cut_edge(self, img):
#         https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #use color image Canny will perform better
        edges = cv2.Canny(img,self.sld1.value(),self.sld2.value())
        
        kernel = np.ones((int(img.shape[0]/10),int(img.shape[1]/10)),np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # kernel = np.ones((100,100),np.uint8)
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Setup SimpleBlobDetector parameters.
        #https://blog.csdn.net/Good_Boyzq/article/details/72811687
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255

        # Filter by Color.
        params.filterByColor = 1
        params.blobColor = 255

        # Filter by Area.
        params.filterByArea = False
        params.minArea = 1500

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.87

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.3

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(closing)
        
        black = np.zeros_like(closing)
        for point in keypoints:
            point.size = self.sld3.value()
            
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
#         #Hough circle
#         print('int(max(np.shape(closing))/4) = ')
#         print(int(max(np.shape(closing))/4))
#         circles = cv2.HoughCircles(im_with_keypoints[:,:,0],cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,
#                                    minRadius=int(max(np.shape(closing))/4) ,maxRadius=int(max(np.shape(closing))))
        
#         if np.shape(circles) is not None:
#             circles = np.uint8(np.around(np.double(circles)))
#             #get the biggest one
#             print(circles)
#             RadiusCircle = np.argmax(circles[0,:,2]) 
#             Circle = circles[0,:][RadiusCircle] 
#              # draw the outer circle
#             im_with_keypoints = cv2.circle(img,(Circle[0],Circle[1]),Circle[2],(50,255,50),2)
#             # draw the center of the circle
#             im_with_keypoints = cv2.circle(img,(Circle[0],Circle[1]),2,(50,255,255),3)
#             print('Circle = ')
#             print(Circle)
        
        im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB)

        
        return im_with_keypoints
        
        
    def cut(self, img, pt1, pt2):
        
        cutImg = img[pt1[1]:pt2[1], pt1[0]:pt2[0],:]
        cutImg = self.cut_edge(cutImg)
#         mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         cutImg = np.array(cutImg)*np.array(mask)
        return cutImg
    
    def paint(self): #draw the img
        self.label_imageDisplay = QLabel()
        pixmap01 = QPixmap.fromImage(self.mQImage)
        pixmap_image = QPixmap(pixmap01)
        height, width, byteValue = self.cvImage.shape
        scale = height/width
#         self.label_imageDisplay.resize(width, height)
#         self.label_imageDisplay.setGeometry(50, 50, 600, 600)
        self.label_imageDisplay.setPixmap(pixmap_image)
        self.label_imageDisplay.setAlignment(Qt.AlignCenter)
        self.label_imageDisplay.setScaledContents(True)
        self.label_imageDisplay.setGeometry(50, 50, 700, 700*scale)
        self.label_imageDisplay.setMinimumSize(1,1)

        # label img area
        scroll = QScrollArea(self, frameWidth=0, frameShape=QScrollArea.NoFrame)
        scroll.setGeometry(50,50,800,600)
        scroll.setWidgetResizable(False)

        scroll.setWidget(self.label_imageDisplay)
        scroll.show()
        

    def keyPressEvent(self, QKeyEvent): #save the img
        super(MyDialog, self).keyPressEvent(QKeyEvent)
        if 's' == QKeyEvent.text():
            cv2.imwrite("result.png", self.cvImage)
        else:
            app.exit(1)
    
if __name__ == '__main__':
    app = QCoreApplication.instance() #加這個和if才不會出錯
    if app is None:
        app = QApplication(sys.argv)
    #新建APP
    ex = App()
    sys.exit(app.exec_())