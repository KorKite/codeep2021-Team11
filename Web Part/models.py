import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow import keras
import numpy as np
from gradcam import Gradcam
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Model:
    def __init__(self):
        self.model = load_model('/home/dsail/codeeplearning/models/oct_model.h5')
        self.gradcam = Gradcam()
        self.label_info = ["CNV", "DME", "DRUSEN", "NORMAL"]

    def get_img_array(self, image_url):
        img = keras.preprocessing.image.load_img(image_url, target_size=(256, 256))
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array
    
    def predict(self, image_url, gradcam_url):
        """
        predict Class using self.model
        """
        img_arr = self.get_img_array(image_url)
        softmax = self.model.predict(img_arr)
        preds = np.argmax(softmax, axis=1)
        preds_val = [round(i, 2) for i in softmax[0]]
        label = self.label_info[int(preds)]
        self.gradcam.get_gradcam(image_url, gradcam_url)

        return {"label":label, "softmax":preds_val}

if __name__ == "__main__":
    model = Model()
    prediction = model.predict()