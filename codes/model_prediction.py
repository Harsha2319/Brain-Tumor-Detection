import os

import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import extract_datafile
from utils import creating_dataframe
from utils import split_dataset
from metrics import dice_coef
from metrics import dice_coef_loss
from metrics import bce_dice_loss
from metrics import iou

app = Flask(__name__)


def create_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

def convert_img_type(image_file_name, img_type):
  image_file_name = image_file_name.replace('/content', '../data')
  img = Image.open(image_file_name)
  rgb_img = img.convert('RGB')
  target_filename = "../static/" + img_type + "/" + image_file_name.split('.')[0].split('/')[-1] +".jpg"
  rgb_img.save(target_filename)

def generate_5_random_img():
  file_name_list = list(os.listdir("../static/MRI"))
  display_images = random.sample(file_name_list, 5)
  display_images = ['MRI/' + f for f in display_images]
  return display_images

def predict(img,mask):
  
  image=plt.imread('../static/'+img)
  image=cv2.resize(image,dsize=(SIZE))
  image=image /255.0
  image_ex=np.expand_dims(image,axis=0)

  pred=new_model.predict(image_ex)
  p=np.reshape(pred,(256,256,1)) 
  pred_img=cv2.threshold(p,0.5,1,cv2.THRESH_BINARY)
  pred_img=np.uint8(pred_img[1])

  filename = mask.replace("mask","pred")
  filepath = "../static/" + filename
  plt.imsave(filepath, pred_img)

  return filename

@app.route("/")
def home():
    global display_images 
    display_images = generate_5_random_img()
    return render_template('index.html', 
                           pic1=display_images[0],
                           pic2=display_images[1],
                           pic3=display_images[2],
                           pic4=display_images[3],
                           pic5=display_images[4])


@app.route('/handle_data', methods=['POST'])
def handle_data():
    image = request.form['image']
    image_link = {'Picture 1':0,
                  'Picture 2':1,
                  'Picture 3':2,
                  'Picture 4':3,
                  'Picture 5':4}
    img = display_images[image_link[image]]           
    mask = "mask/" +img.split('.')[0].split('/')[1] + "_mask.jpg"
    pred = predict(img,mask)
    return render_template('output.html', 
                           pic1=display_images[0],
                           pic2=display_images[1],
                           pic3=display_images[2],
                           pic4=display_images[3],
                           pic5=display_images[4],
                           img=img,
                           mask = mask,
                           pred=pred)
    
dataset_path = "../data/Brain_MRI_Data.zip"
dir_extracted = "../data/lgg-mri-segmentation/"
model_path = "../model/BrainTumorDetection.h5"
SIZE = (256, 256)
display_images = [],

extract_datafile(dataset_path, dir_extracted)
df = creating_dataframe()
df_train, df_val, df_test = split_dataset(df)

create_dir("../static")
create_dir("../static/MRI")
create_dir("../static/mask")
create_dir("../static/pred")

for filename in df_test['image']:
  convert_img_type(filename, "MRI")
for filename in df_test['mask']:
  convert_img_type(filename, "mask")

new_model = tf.keras.models.load_model(model_path, custom_objects={"bce_dice_loss": bce_dice_loss, "iou":iou, "dice_coef":dice_coef})

app.run()

