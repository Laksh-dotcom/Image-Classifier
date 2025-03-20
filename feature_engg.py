import numpy as np
import pywt
import cv2    
import os
import matplotlib.pyplot as plt
import joblib as jb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def w2d(img, mode='haar', level=1):
    # Convert to grayscale
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to float
    imArray = np.float32(imArray) / 255.0  

    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients (Set Approximation Coefficients to 0)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255.0  

    # Clip values to range (0-255) before converting to uint8
    imArray_H = np.clip(imArray_H, 0, 255).astype(np.uint8)

    return imArray_H
"""
FURTHER THIS IS WHERE I CONVERT THE GIVEN IMAGE INTO THE WAVELET FORM...



"""
def main_func():
    celebrity_dict = {}
    file_path = 'F:/Coding/Python/ML/Image Classifier/data_set/cropped'

    celebrity_dict = {os.path.basename(i.path):[] for i in os.scandir(file_path) if i.is_dir()}
    img_dir = [i.path for i in os.scandir(file_path) if i.is_dir()]
    for i in img_dir:
        for j in os.scandir(i):
            celebrity_dict[os.path.basename(i)].append(j.path)
    x = []
    y = []
    count = 0
    class_dict = {}
    for i in celebrity_dict.keys():
        class_dict[i] = count
        count += 1
    print(class_dict)
    for c_n, t_f in celebrity_dict.items():
        for t_fs in t_f:
            img = cv2.imread(t_fs)
            scaled_img_raw = cv2.resize(img, (32, 32))

            img_har = w2d(img, 'db1', 5)
            scaled_har_img = cv2.resize(img_har, (32, 32))

            combined = np.vstack((scaled_img_raw.reshape(32*32*3, 1), scaled_har_img.reshape(32*32, 1)))
            x.append(combined)
            y.append(class_dict[c_n])
    return x, y
"""
THIS COMING IS GOING TO THE PART WHEREIN I WILL TRAIN MY DATA...



"""
def training(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    return x_test, model, model.score(x_test, y_test)
x, y = main_func()
x = np.array(x).reshape(len(x), -1)
x_test, mdl, ans = training(x, y)
print(ans)
"""
SAVING THE MODEL IN A FILE....


"""
jb.dump(mdl, 'F:/Coding/Python/ML/Image Classifier/Image_Classifier.pkl')

mdl_loaded = jb.load('F:/Coding/Python/ML/Image Classifier/Image_Classifier.pkl')

y_pred = mdl_loaded.predict(x_test)

print(y_pred)