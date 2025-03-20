import cv2
import matplotlib.pyplot as plt
import os
import shutil
class DetectFaceAndEyes:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def convert_2_zoom_if_eyes_present(self, imgpath):
        img = cv2.imread(imgpath)
        if img is None:
            print(f"Error: Image not found or invalid format -> {imgpath}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                return roi_color  
        
        return None  

class CreateCroppedFolder:
    def __init__(self):
        self.path_to_dataset = 'F:\Coding\Python\ML\Image Classifier\data_set'
        self.crpfolder = 'F:\Coding\Python\ML\Image Classifier\data_set\cropped'
    
    def print_list(self):
        self.img_dir = [i.path for i in os.scandir(self.path_to_dataset) if i.is_dir()]
        return self.img_dir
    
    def create_cropped_folder(self):
        obj = DetectFaceAndEyes()
        
        if os.path.exists(self.crpfolder):
            shutil.rmtree(self.crpfolder)
        os.mkdir(self.crpfolder)

        self.print_list()
        crp_img_dir = []

        for folder in self.img_dir:
            name = os.path.basename(folder)
            count = 1  # Move count outside the loop
            
            for img_file in os.scandir(folder):
                roi_color = obj.convert_2_zoom_if_eyes_present(img_file.path)  # Pass file path
                
                if roi_color is not None:
                    crop_folder = os.path.join(self.crpfolder, name)
                    if not os.path.exists(crop_folder):
                        os.mkdir(crop_folder)
                        crp_img_dir.append(crop_folder)

                    cropped_file = f"{name}_{count}.png"
                    cropped_file_path = os.path.join(crop_folder, cropped_file)
                    
                    cv2.imwrite(cropped_file_path, roi_color)
                    count += 1

obj1 = CreateCroppedFolder()
print(obj1.print_list())
obj1.create_cropped_folder()
