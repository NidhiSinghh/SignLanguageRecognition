import cv2
import numpy as np
import os
import string
# Create the directory structure
if not os.path.exists("data2"):
    os.makedirs("data2")
    os.makedirs("data2/train")
    os.makedirs("data2/test")
    os.makedirs("data2/train/0")
    os.makedirs("data2/train/1")
    os.makedirs("data2/train/2")
    os.makedirs("data2/train/3")
    os.makedirs("data2/train/4")
    os.makedirs("data2/train/5")
    os.makedirs("data2/test/0")
    os.makedirs("data2/test/1")
    os.makedirs("data2/test/2")
    os.makedirs("data2/test/3")
    os.makedirs("data2/test/4")
    os.makedirs("data2/test/5")
    os.makedirs("data2/train/6")
    os.makedirs("data2/train/7")
    os.makedirs("data2/train/8")
    os.makedirs("data2/train/9")
    os.makedirs("data2/test/6")
    os.makedirs("data2/test/7")
    os.makedirs("data2/test/8")
    os.makedirs("data2/test/9")
    os.makedirs("data2/train/A")
    os.makedirs("data2/train/B")
    os.makedirs("data2/train/C")
    os.makedirs("data2/train/D")
    os.makedirs("data2/train/E")
    os.makedirs("data2/train/F")
    os.makedirs("data2/train/G")
    os.makedirs("data2/train/H")
    os.makedirs("data2/train/I")
    os.makedirs("data2/train/J")
    os.makedirs("data2/train/K")
    os.makedirs("data2/train/L")
    os.makedirs("data2/train/M")
    os.makedirs("data2/train/N")
    os.makedirs("data2/train/O")
    os.makedirs("data2/train/P")
    os.makedirs("data2/train/Q")
    os.makedirs("data2/train/R")
    os.makedirs("data2/train/S")
    os.makedirs("data2/train/T")
    os.makedirs("data2/train/U")
    os.makedirs("data2/train/V")
    os.makedirs("data2/train/W")
    os.makedirs("data2/train/X")
    os.makedirs("data2/train/Y")
    os.makedirs("data2/train/Z")




    os.makedirs("data2/test/A")
    os.makedirs("data2/test/B")
    os.makedirs("data2/test/C")
    os.makedirs("data2/test/D")
    os.makedirs("data2/test/E")
    os.makedirs("data2/test/F")
    os.makedirs("data2/test/G")
    os.makedirs("data2/test/H")
    os.makedirs("data2/test/I")
    os.makedirs("data2/test/J")
    os.makedirs("data2/test/K")
    os.makedirs("data2/test/L")
    os.makedirs("data2/test/M")
    os.makedirs("data2/test/N")
    os.makedirs("data2/test/O")
    os.makedirs("data2/test/P")
    os.makedirs("data2/test/Q")
    os.makedirs("data2/test/R")
    os.makedirs("data2/test/S")
    os.makedirs("data2/test/T")
    os.makedirs("data2/test/U")
    os.makedirs("data2/test/V")
    os.makedirs("data2/test/W")
    os.makedirs("data2/test/X")
    os.makedirs("data2/test/Y")
    os.makedirs("data2/test/Z")
    

# Train or test 
mode = 'train'
directory = '/home/nidhik/signlanguagedirectory/code/homepetry/data2/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {
            'zero': len(os.listdir(directory+"/0")),
            'one': len(os.listdir(directory+"/1")),
            'two': len(os.listdir(directory+"/2")),
            'three': len(os.listdir(directory+"/3")),
            'four': len(os.listdir(directory+"/4")),
            'five': len(os.listdir(directory+"/5")),
            'six': len(os.listdir(directory+"/6")),
            'seven': len(os.listdir(directory+"/7")),
            'eight': len(os.listdir(directory+"/8")),
            'nine': len(os.listdir(directory+"/9")),
             'a': len(os.listdir(directory+"/A")),
             'b': len(os.listdir(directory+"/B")),
             'c': len(os.listdir(directory+"/C")),
             'd': len(os.listdir(directory+"/D")),
            #  'e': len(os.listdir(directory+"/E")),
            #  'f': len(os.listdir(directory+"/F")),
             'g': len(os.listdir(directory+"/G")),
            #  'h': len(os.listdir(directory+"/H")),
            #  'i': len(os.listdir(directory+"/I")),
             'j': len(os.listdir(directory+"/J")),
             'k': len(os.listdir(directory+"/K")),
             'l': len(os.listdir(directory+"/L")),
            #  'm': len(os.listdir(directory+"/M")),
            #  'n': len(os.listdir(directory+"/N")),
             'o': len(os.listdir(directory+"/O")),
             'p': len(os.listdir(directory+"/P")),
             'q': len(os.listdir(directory+"/Q")),
            #  'r': len(os.listdir(directory+"/R")),
             's': len(os.listdir(directory+"/S")),
             't': len(os.listdir(directory+"/T")),
             'u': len(os.listdir(directory+"/U")),
            #  'v': len(os.listdir(directory+"/V")),
             'w': len(os.listdir(directory+"/W")),
             'x': len(os.listdir(directory+"/X")),
             'y': len(os.listdir(directory+"/Y")),
             'z': len(os.listdir(directory+"/Z"))


             
            }
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SIX : "+str(count['six']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SEVEN : "+str(count['seven']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "EIGHT : "+str(count['eight']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "NINE : "+str(count['nine']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "a : "+str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "b : "+str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "c : "+str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "d : "+str(count['d']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "e : "+str(count['e']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "f : "+str(count['f']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "g : "+str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "h : "+str(count['h']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "i : "+str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "k : "+str(count['k']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "l : "+str(count['l']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "m : "+str(count['m']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "n : "+str(count['n']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "o : "+str(count['o']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "p : "+str(count['p']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "q : "+str(count['q']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "r : "+str(count['r']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "s : "+str(count['s']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "t : "+str(count['t']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "u : "+str(count['u']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "v : "+str(count['v']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "w : "+str(count['w']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "x : "+str(count['x']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "y : "+str(count['y']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "z : "+str(count['z']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    
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
    roi = cv2.resize(roi, (310, 310)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
    
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'7/'+str(count['seven'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'8/'+str(count['eight'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'9/'+str(count['nine'])+'.jpg', roi)
    
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['d'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('e'):
    #     cv2.imwrite(directory+'E/'+str(count['e'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('f'):
    #     cv2.imwrite(directory+'F/'+str(count['f'])+'.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['g'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('h'):
    #     cv2.imwrite(directory+'H/'+str(count['h'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('i'):
    #     cv2.imwrite(directory+'I/'+str(count['i'])+'.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['j'])+'.jpg', roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'K/'+str(count['k'])+'.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['l'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('m'):
    #     cv2.imwrite(directory+'M/'+str(count['m'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('n'):
    #     cv2.imwrite(directory+'N/'+str(count['n'])+'.jpg', roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'O/'+str(count['o'])+'.jpg', roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'P/'+str(count['p'])+'.jpg', roi)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'Q/'+str(count['q'])+'.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'R/'+str(count['r'])+'.jpg', roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['s'])+'.jpg', roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'T/'+str(count['t'])+'.jpg', roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['u'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('v'):
    #     cv2.imwrite(directory+'V/'+str(count['v'])+'.jpg', roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'W/'+str(count['w'])+'.jpg', roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'X/'+str(count['x'])+'.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Y/'+str(count['y'])+'.jpg', roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Z/'+str(count['z'])+'.jpg', roi)       
    
    
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data2/test/0"
newd = "data2/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""
