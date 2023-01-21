import cv2
import numpy as np
from keras.models import load_model

#tf 2.9.1
#keras 2.6.0

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

model = load_model('Keras_model.h5',compile=False)
data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)
classes = ["1 real","25 cent","50 cent"]

def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

def DetectarMoeda(img):
    imgMoeda = cv2.resize(img,(224,224))
    imgMoeda = np.asarray(imgMoeda)
    imgMoedaNormalize = (imgMoeda.astype(np.float32)/127.0)-1
    data[0] = imgMoedaNormalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    return classe,percent



while True:
    _,img = video.read()
    img = cv2.resize(img,(640,480))
    imgPre = preProcess(img)
    countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    qtd = 0
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            recorte = img[y:y +h,x:x+ w]
            classe, conf = DetectarMoeda(recorte)
            if conf >0.7:
                cv2.putText(img,str(classe),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                if classe == '1 real': qtd+=1
                if classe == '25 cent': qtd += 0.25
                if classe == '50 cent': qtd += 0.5

    cv2.rectangle(img,(430,30),(600,80),(0,0,255),-1)
    cv2.putText(img,f'R$ {qtd}',(440,67),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

    cv2.imshow('IMG',img)
    cv2.imshow('IMG PRE', imgPre)
    cv2.waitKey(1)