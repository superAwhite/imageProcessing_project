import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as plt

plt.use("Qt5Agg")  # 声明使用QT5

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QLabel, QPushButton, QGraphicsView, QGraphicsScene
# , QFileDialog, QApplication 
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication ,Qt
from PyQt5.QtGui import QPixmap, QImage, qRgb, QFont, qRed, qGreen, qBlue, QColor
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *

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
        img = np.array(self.ImgForCut)
        object_list = list(self.label_dict.keys())
        self.Object_scrollArea = QScrollArea(self)
        self.Object_scrollArea.setGeometry(700,150,300,600)
        
        
        for obj in object_list:
            print(obj)
            _, pt1, pt2 = self.label_dict[obj]
            cutImg = self.cut(img, pt1, pt2)
            print(np.shape(img))
            print(np.shape(cutImg))
            cutImg_cv2 = cv2.cvtColor(np.array(cutImg), cv2.COLOR_RGB2BGR)
            cv2.imshow(obj, cutImg_cv2)
#             self.displayEachObject_addEachRow()
            
#         self.Object_scrollArea.addStretch(2)
#         self.Object_scrollArea.show()
        
    def displayEachObject_addEachRow(self,name, cutImg,prob, pt1, pt2): 
        layout = QHBoxLayout()
        #img column
        label_img = QLabel()        
        pixmap = QPixmap.fromImage(cutImg)
        pixmap_cutImg = QPixmap(pixmap)
        label_img.setPixmap(pixmap_cutImg)
        label_img.setAlignment(Qt.AlignCenter)
        label_img.setGeometry(50,50)
        label_img.setScaledContents(True)     
        layout.addWidget(label_img)
        #item name column
        label_item = QLabel()
        label_item.setText('Name\nProbability\nRegion')
        layout.addWidget(label_item)
        #value column
        label_value = QLabel()
        label_value.setText(name+'\n'+prob+'\n'+str(pt1)+' '+str(pt2))
        layout.addWidget(label_value)
        #region width
        layout.addStretch(1)
        #add to scroll area
        self.Object_scrollArea.addWidget(layout)
        
        
    def cut(self, img, pt1, pt2):
        print(pt1,pt2)
        cutImg = img[pt1[1]:pt2[1], pt1[0]:pt2[0],:]
        
        return cutImg
    
    def paint(self): #draw the img
        self.label_imageDisplay = QLabel()
        pixmap01 = QPixmap.fromImage(self.mQImage)
        pixmap_image = QPixmap(pixmap01)
        height, width, byteValue = self.cvImage.shape
#         scale = 700/width
#         self.label_imageDisplay.resize(width, height)
#         self.label_imageDisplay.setGeometry(50, 50, 600, 600)
        self.label_imageDisplay.setPixmap(pixmap_image)
        self.label_imageDisplay.setAlignment(Qt.AlignCenter)
        self.label_imageDisplay.setScaledContents(True)
        self.label_imageDisplay.setMinimumSize(1,1)

        
        scroll = QScrollArea(self, frameWidth=0, frameShape=QScrollArea.NoFrame)
        scroll.setGeometry(50,50,800,600)
        scroll.setWidgetResizable(False)
#         scroll.setFixedWidth(800)
        scroll.setWidget(self.label_imageDisplay)
        scroll.show()
        

    def keyPressEvent(self, QKeyEvent): #save the img
        super(MyDialog, self).keyPressEvent(QKeyEvent)
        if 's' == QKeyEvent.text():
            cv2.imwrite("cat2.png", self.cvImage)
        else:
            app.exit(1)
    
if __name__ == '__main__':
    app = QCoreApplication.instance() #加這個和if才不會出錯
    if app is None:
        app = QApplication(sys.argv)
    #新建APP
    ex = App()
    sys.exit(app.exec_()) 