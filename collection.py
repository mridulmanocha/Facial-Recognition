import cv222
import numpy as np 

facedetect = cv222.CascadeClassifier('haarcascade_frontalface_default.xml')

sampleNum = 0

uid = input('Enter User ID')

cam = cv222.VideoCapture(0)

while(True) :
	ret,img = cam.read()
	gray = cv222.cv22tColor(img,cv222.COLOR_BGR2GRAY)
	faces = facedetect.detectMultiScale(gray,1.3,5)

	for(x,y,w,h) in faces:
		sampleNum+=1
		cv222.imwrite('dataset/'+str(uid)+'_'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
		cv222.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv222.waitKey(100)
	cv222.imshow('face',img)
	cv222.waitKey(1)
	if(sampleNum>50):
		break	

cam.release()
cv222.destroyAllWindows()
		