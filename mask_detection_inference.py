from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QThread,pyqtSignal
import sys
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
from tensorflow.image import resize,ResizeMethod
import numpy as np
import torch
from Yolo.detect_face import get_faces_bbox
from Yolo.models.experimental import attempt_load

BASE_DIR = os.path.abspath(os.getcwd())
TOLERANCE = 0.1
IMAGE_SHAPE = (128,128)

class Ui(QtWidgets.QMainWindow):
    img = None
    camState = False
    mask_model = load_model(os.path.join(BASE_DIR, 'mask_model\cnn-rgb 128.h5'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    	
    yolo_priority = ['yolov5n-face.pt','yolov5s-face.pt','yolov5m-face.pt','yolov5l-face.pt']
    available_yolo = [check_yolo for check_yolo in os.listdir(os.path.join(BASE_DIR, 'Yolo')) if check_yolo.endswith('.pt')]

    print("Available Yolo : ",available_yolo)
    for yolo in yolo_priority:
        if yolo in available_yolo:
            print("Using Yolo : ",yolo)
            yolo_model = attempt_load(os.path.join(BASE_DIR, 'Yolo',yolo), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            break

    assert yolo_model is not None, "No Yolo model found"

    # yolo_model = attempt_load("Yolo\yolov5n-0.5.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('new_DL.ui', self)
        # # this will hide the title bar
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
  
        # # set the title
        # self.setWindowTitle("no title")

        self.stack_pages = self.findChild(QtWidgets.QStackedWidget, 'stackedWidget')
        self.stack_pages.setCurrentIndex(0)
        self.btn_page1 = self.findChild(QtWidgets.QPushButton, 'btn_page1')
        self.btn_page1.clicked.connect(lambda: self.changePage(0))

        self.btn_page2 = self.findChild(QtWidgets.QPushButton, 'btn_page2')
        self.btn_page2.clicked.connect(lambda: self.changePage(1))
        
        self.btnOpenFile = self.findChild(QtWidgets.QPushButton, 'btn_select_img')
        self.btnOpenFile.clicked.connect(self.loadImage)

        self.btnOpenCam = self.findChild(QtWidgets.QPushButton, 'btn_opencam')
        self.btnOpenCam.clicked.connect(self.setCam)

        self.image_ui = self.findChild(QtWidgets.QLabel, 'img_detect')

        self.camera_ui = self.findChild(QtWidgets.QLabel, 'camera_detect')

        self.cam_port = self.findChild(QtWidgets.QComboBox, 'port_selector')
        self.cam_port.addItems(self.DetectCamPort())
        self.cam_port.setCurrentIndex(0)
        self.cam_port.activated.connect(self.clearUI)

        self.camWorker = CamWorker()
        self.camWorker.setModel(self.mask_model,self.yolo_model,self.device)

        self.setupTensorflow()
        self.show()

    def changePage(self, page = 0):
        print("Change Page to : ",page)
        self.stack_pages.setCurrentIndex(page)

    def setupTensorflow(self):
        pred_image = np.zeros((IMAGE_SHAPE[0],IMAGE_SHAPE[1],3),dtype=np.uint8)
        pred_image = pred_image/255.0
        pred_image = expand_dims(pred_image, axis=0)
        pred = np.argmax(self.mask_model.predict(pred_image), axis=1)

    def clearUI(self):
        self.camera_ui.clear()

    def DetectCamPort(self):
        self.valid_cams = []
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                print('Found video source at : ', i)
                self.valid_cams.append(str(i))
            cap.release()
        return self.valid_cams

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.showPrediction()
        return super().resizeEvent(a0)

    def tabChanged(self):
        print("Tab Changed to : ",self.tabWidget.currentIndex())

    def rezizeandShow(self,img,ui):  
        img_temp = img.copy()
        try:
            y,x,color = img_temp.shape
        except:
            y,x = img.shape
            color = 1
        
        if(y>x):
            cv2.rotate(img_temp, cv2.ROTATE_90_CLOCKWISE)

        bytesPerLine = 3 * x
        if(color== 1):
            _pmap = QImage(img_temp, x, y, x, QImage.Format_Grayscale8)
        else:
            _pmap = QImage(img_temp, x, y, bytesPerLine, QImage.Format_RGB888)
        _pmap = QPixmap(_pmap)

        (width,height) = ui.size().width(),ui.size().height()
        imgresized = _pmap.scaled(width, height, QtCore.Qt.IgnoreAspectRatio)

        # test = _pmap.toImage()
        # test.pixel(0,0)
        # for i in range(y):
        #     for j in range(x):
        #         print(f"{i,j} : {QtGui.qRed(test.pixel(i,j))},{QtGui.qGreen(test.pixel(i,j))},{QtGui.qBlue(test.pixel(i,j))}")
        
        
        return imgresized

    def setCam(self):
        if(self.camState):
            print("Camera Closed")
            self.camState = False
            self.btnOpenCam.setText("Nyalakan Kamera")
            self.camWorker.stop()
        else:
            print("Camera Opened")
            try:
                self.camState = True
                self.btnOpenCam.setText("Matikan Kamera")
                self.camWorker.setPort(int(self.cam_port.currentText()))
                self.camWorker.start()
                self.camWorker.ImageSignal.connect(self.updateCamUI)
            except:
                self.camState = False
                self.btnOpenCam.setText("Nyalakan Kamera")
                print("Error Opening Camera")
                self.camWorker.stop()
                self.updateCamUI(np.zeros((480,640,3),dtype=np.uint8))

    def updateCamUI(self,img):
        self.camera_ui.setPixmap(self.rezizeandShow(img,self.camera_ui))
        

    def loadImage(self):
        #hack
        if self.img is not None:
            try:
                _tmp_img = self.img
            except:
                pass
        # This is executed when the button is pressed
        print('btn Open File Pressed')
        openFileDialog = QFileDialog.getOpenFileName(self,"select Image File",os.getcwd(),"Image Files (*.jpg *.gif *.bmp *.png *.tiff *.jfif)")
        fileimg = openFileDialog[0]

        try:
            if(len(fileimg)>4):
                self.img = cv2.imread(fileimg)
                rgb_im = BGR2RGB(self.img.copy())

                prediction_img = rgb_im.copy()

                imageHeight, imageWidth, _ = prediction_img.shape
                scale = 0.1

                faces = get_faces_bbox(rgb_im,self.yolo_model,self.device)
                for idx,face in enumerate(faces):
                    face_start_point = int(face[0]*(1-TOLERANCE)),int(face[1]*(1-TOLERANCE))
                    face_end_point = int(face[2]*(1+TOLERANCE)),int(face[3]*(1+TOLERANCE))
                    cropped_image = rgb_im[face_start_point[1]:face_end_point[1],face_start_point[0]:face_end_point[0]]
                
                    pred_image = resize(cropped_image, [IMAGE_SHAPE[0],IMAGE_SHAPE[1]], method=ResizeMethod.BICUBIC)
                    pred_image = pred_image/255.0
                    pred_image = expand_dims(pred_image, axis=0)
                    pred = np.argmax(self.mask_model.predict(pred_image), axis=1)

                    label,color = Num2Label(pred[0])
                    cv2.rectangle(prediction_img, face_start_point, face_end_point, color, 2)
                    cv2.putText(prediction_img, label, face_start_point, cv2.FONT_HERSHEY_PLAIN, min(imageWidth,imageHeight)/(20/scale), color, 2, cv2.LINE_AA)

                self.img_prediction = prediction_img
                self.showPrediction()
        except:
            try:
                self.img = _tmp_img
            except:
                pass
            QMessageBox.about(self, "Error", "Gagal Memuat Gambar")
    
    def showPrediction(self):
        if(self.img is not None):
            img_inference = self.img_prediction.copy()
            # img_inference = BGR2RGB(img_inference)
            self.image_ui.setPixmap(self.rezizeandShow(img_inference,self.image_ui))


class CamWorker(QThread):
    ImageSignal = pyqtSignal(np.ndarray)
    port = 0
    mask_model = None
    cam = None

    def run(self):
        self.ThreadActive = True
        self.scale = 0.1
        while self.ThreadActive:
            success, image = self.cam.read()
            image = cv2.flip(image, 1)
            imageHeight, imageWidth, _ = image.shape
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prediction_img = image.copy()
            
            faces = get_faces_bbox(prediction_img,self.yolo_model,self.device)
            if faces is not None and len(faces)>0:
                for idx,face in enumerate(faces):
                    face_start_point = int(face[0]*(1-TOLERANCE)),int(face[1]*(1-TOLERANCE))
                    face_end_point = int(face[2]*(1+TOLERANCE)),int(face[3]*(1+TOLERANCE))
                    cropped_image = prediction_img[face_start_point[1]:face_end_point[1],face_start_point[0]:face_end_point[0]]
                
                    pred_image = resize(cropped_image, [IMAGE_SHAPE[0],IMAGE_SHAPE[1]], method=ResizeMethod.BICUBIC)
                    pred_image = pred_image/255.0
                    pred_image = expand_dims(pred_image, axis=0)
                    pred = np.argmax(self.mask_model.predict(pred_image), axis=1)

                    label,color = Num2Label(pred[0])
                    cv2.rectangle(prediction_img, face_start_point, face_end_point, color, 2)
                    cv2.putText(prediction_img, label, face_start_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, min(imageWidth,imageHeight)/(20/self.scale), color, 2, cv2.LINE_AA)

                self.ImageSignal.emit(prediction_img)
            else:
                self.ImageSignal.emit(image)
        self.ImageSignal.emit(np.zeros([512,512,3],np.uint8))

    def setModel(self,mask_model,yolo_model,device):
        self.mask_model = mask_model
        self.yolo_model = yolo_model
        self.device = device

    def setPort(self,port):
        self.port = port
        if self.cam is not None and self.cam.isOpened():
            self.cam.release()
        self.cam = cv2.VideoCapture(self.port)
    
    def stop(self):
        self.ImageSignal.emit(np.zeros([512,512,3],np.uint8))
        self.ThreadActive = False
        self.cam.release()
        self.quit()

def BGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def Num2Label(num):
    if num == 0:
        return "incorrect_mask",(255, 255, 0)
    elif num == 1:
        return "with_mask",(0, 255, 0)
    elif num == 2:
        return "without_mask",(255, 0, 0)
    else:
        raise ValueError("Invalid number")

app = QtWidgets.QApplication(sys.argv)
window = Ui()
inference_app = app.exec_()
sys.exit(inference_app)