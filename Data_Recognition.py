import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context',None)) : 
    ssl._create_default_https_context = ssl._create_unverified_context
# Here we are going to be using a dataset from OpenML where we have 70,000 images of hand-written digits.

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
# code to fit the data to the model.
#  we were dealing with binary logistic regressions, but here, we have 10 labels, 0 to 9. 
# For this, we write multi_class='multinomial' to specify that this is a multinomial logistic regression.
clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled, y_train)
#  there is a solver involved in all the logistic regressions, and the default solver is liblinear, which is highly efficient for linear logistic regression. This is also efficient with
# binary logistic regressions that we learned earlier. For multinomial logistic regression, solver='saga' is highly efficient. It works well with a large number of samples and 
# supports multinomial logistic regressions

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# start the camera
capture = cv2.VideoCapture(0)

while(True) : 
    try:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # DRAW A BOX IN THE CENTER OF THE VIDEO
        height, width = gray.shape
        ULeft = (int(width/2-50), int(height/2-50))
        BRight = (int(width/2+50), int(height/2+50))
        cv2.rectangle(gray, ULeft, BRight, (0, 255, 0),2)
        # TO FOCUS ON THE AREA IN THE BOX FOR DETECTING THE DIGIT
        # roi = REGION OF INTEREST
        roi = gray[ULeft[1] : BRight[1], ULeft[0] : BRight[0]]
        #CONVERT CV2 INTO PIL FORMAT
        # PIL = PYTHON IMAGING LIBRARY
        img_pil = Image.fromarray(roi)
        
        img_bw = img_pil.convert('L')
        img_bw_resize = img_bw.resize((28,28), Image.ANTIALIAS)
        img_bw_resize_inverted = PIL.ImageOps.invert(img_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resize, pixel_filter)
        img_bw_resize_inverted_scaled = np.clip(img_bw_resize_inverted-min_pixel, 0, 255)
        max_pixel = np.max(img_bw_resize_inverted)
        img_bw_resize_inverted_scaled = np.asarray(img_bw_resize_inverted_scaled)/max_pixel
        tSample = np.array(img_bw_resize_inverted_scaled).reshape(1,784)
        tPredict = clf.predict(tSample)
        print("Printed Class: ", tPredict)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') : 
            break
    
    except Exception as e : 
        pass

capture.release()
cv2.destroyAllWindows()