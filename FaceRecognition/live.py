

import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
left_eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")


while(True):
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
		roi_gray = gray[ y:y+h, x:x+w]
		roi_color = frame[ y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
		left = left_eye_cascade.detectMultiScale(roi_gray)
		right = right_eye_cascade.detectMultiScale(roi_gray)
		#for (rx,ry,rw,rh),(lx,ly,lw,lh) in zip(right,left):
			#if ((rx+rw)-rx) == ((lx+lw)-lx):
		for (rx,ry,rw,rh) in right:
			for (lx,ly,lw,lh) in left:
				cv2.rectangle(roi_color, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
				print("Right:({0},{1})and({0},{1})",rx,rw,ry,rh)
				cv2.rectangle(roi_color, (lx, ly), (lx+rw, ly+rh), (0, 0, 255), 2)
				print("Left:({0},{1})and({0},{1})",lx,lw,ly,lh)




	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
