model=load_model("/content/facerecognition.h5")
import cv2
photo = cv2.read("photo_path")
photo=cv2.resize(photo,(299,299))
fullbody_classifier=cv2 .CascadeClassifier("haarcascade_frontalface_default.xml")
faces=fullbody_classifier.detectMultiScale(photo,1.3,2)
for x,y,w,h in faces:
    face = photo[y:y+h, x:x+w]
    if type(face) is np.ndarray:
      face = cv2.resize(face, (299,299))
      im = Image.fromarray(face, 'RGB')
      img_array = np.array(im)
      img_array = np.expand_dims(img_array, axis=0)
      pred = model.predict(img_array)          
      name=""
      if(pred[0][0]>0.5):
        name='NAME1'
        cv2.putText(photo,name,(x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow(photo)

      elif(pred[0][1]>0.5):
        name='NAME2'
        cv2.putText(photo,name,(x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow(photo)

    else:
      print("No face found")


    cv2.destroyAllWindows()
