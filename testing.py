import cv222
import numpy as np 

facedetect = cv222.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv222.face.LBPHFaceRecognizer_create()
rec.read('trainingData.yml')


id = 0 
fontFace = cv222.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,255)

id_map = ['Mridul','Gurdeep']

cam = cv222.VideoCapture(0)

while(True):
	ret,img = cam.read()
	gray = cv222.cv22tColor(img,cv222.COLOR_BGR2GRAY)
	faces = facedetect.detectMultiScale(gray,1.3,5)

	for(x,y,w,h) in faces:
		cv222.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		id,conf = rec.predict(gray[y:y+h,x:x+w])

		cv222.putText(img,str(id_map[id-1])+'_'+str(conf),(x,y+h),fontFace,fontScale,fontColor)

	cv222.imshow("face",img)
	if(cv222.waitKey(1)==ord('q')):
		break

cam.release()
cv222.destroyAllWindows()