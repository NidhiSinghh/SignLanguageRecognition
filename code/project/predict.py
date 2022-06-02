import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

json_file = open("/home/nidhik/signlanguagedirectory/code/project/model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("/home/nidhik/signlanguagedirectory/code/project/model-bw.h5")
print("Loaded model from disk")

#cap json_file = open("model-bw.json", "r")
cap=cv2.VideoCapture(0)
# Category dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE',6:'SIX',7:'SEVEN',8:'EIGHT',9:'NINE',
              10:"A",11:'B',12:'C',13:'D',14:'G',15:'J',16:'K',17:'L',18:'O',19:'P',20:'Q',21:'S',22:'T',23:'U',24:'W',25:'X',26:'Y',27:'Z'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    # = cv2.VideoCapture(0)
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi,100, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    # print(result)
    prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'SIX':result[0][6],
                  'SEVEN':result[0][7],
                  'EIGHT':result[0][8],
                  'NINE':result[0][9],
                  'A':result[0][10],
                  'B':result[0][11],
                  
                  
                  'C':result[0][12],
                  'D':result[0][13],
                  'G':result[0][14],
                  'J':result[0][15],
                  'K':result[0][16],
                  'L':result[0][17],
                  'O':result[0][18],
                  'P':result[0][19],
                  'Q':result[0][20],
                  'S':result[0][21],
                  'T':result[0][22],
                  'U':result[0][23],
                  'W':result[0][24],
                  'X':result[0][25],
                  'Y':result[0][26],
                  'Z':result[0][27]
    }

    #print(prediction)
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    #print(prediction)

    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (x1+200,y2+20), cv2.FONT_HERSHEY_PLAIN, 2
    , (0,0,255),2, 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
