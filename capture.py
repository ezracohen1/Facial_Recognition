import cv2, time, pandas
from datetime import datetime
captureDevice = cv2.VideoCapture(0)
check, frame=captureDevice.read()

face_cascade=cv2.CascadeClassifier(r"C:\Users\ezrat\OneDrive\Documents\GitHub\Facial_Recognition\haarcascade_frontalface_default.xml")
status=0
status_list=[]
#times=[]
#df=pandas.DataFrame(columns=["Start", "End"])
while True:
    check, frame=captureDevice.read()
    #print(check)
    #print(frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.35, minNeighbors=5)
    for x, y, w, h in faces:
        img=cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0),3)
    status=len(faces)
    #status_list.append(status)
    #if len(status_list)>1:
    #    if status_list[-1]==1 and status_list[-2]==0:
    #        times.append(datetime.now())
    #    if status_list[-1]==0 and status_list[-2]==1:
    #        times.append(datetime.now())
    cv2.imshow('Capturing', frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        #if status==1:
        #    times.append(datetime.now())
        break
#if len(times)!=1:
#    for i in range(0,len(times),2):
#        df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)
#df.to_csv('times.csv')
captureDevice.release()
cv2.destroyAllWindows()