import os
import zipfile
from glob import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

SIZE = (256, 256)
BATCH_SIZE = 16

def extract_datafile(dataset_path, dir_extracted, extract_to):
  
  """
  Extracts dataset for .zip file.

  Args:
    dataset_path: Path to the location of dataset
    dir_extracted: Path to directory extracted from the dataset
  """
  
  if os.path.isdir(dir_extracted):
    print("Dataset already extracted...")
  else:
    zfile = zipfile.ZipFile(dataset_path)
    zfile.extractall(extract_to)


def creating_dataframe():
  
  """
  creates dataframe with 2 cols: input filenames (features), mask filenames (target)

  Returns:
    df: dataframe containing input filenames & mask filenames
  """
  
  # Getting list of all mask file names using the pattern
  pattern = "../data/lgg-mri-segmentation/kaggle_3m/*/*_mask*"
  mask_files = glob(pattern)

  # Generating a list of all training file names with brain MRI images
  train_files = [file.replace('_mask','') for file in mask_files]

  df = pd.DataFrame({"image": train_files,
                   "mask": mask_files})
  return df


def split_dataset(df):
  
  """
  Splits input dataframe into 3 dataframe: 90% data in training, 5% data for validating & 5% data for testing the model 

  Args
    df: dataframe containing brain MRI filenames & mask filenames

  Returns:
    df_train: Dataframe used to train the model
    df_val: Dataframe used to validate the model
    df_test: Dataframe used to test the model
  """
  
  df_train, df_test = train_test_split(df, test_size=0.1)
  df_test, df_val = train_test_split(df_test, test_size=0.5)
  return df_train, df_val, df_test

def adjust_data(img,mask):
  
  """


  Args:


  Returns:

  """
  
  img = img / 255.
  mask = mask / 255.
  mask[mask > 0.5] = 1
  mask[mask <= 0.5] = 0

  return (img, mask)

def batch_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=SIZE,
                    seed=1):

  """


  Args:


  Returns:

  """
  
  image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
  mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    
  image_generator = image_datagen.flow_from_dataframe(data_frame,
                                                      x_col = "image",
                                                      class_mode = None,
                                                      color_mode = image_color_mode,
                                                      target_size = target_size,
                                                      batch_size = batch_size,
                                                      save_to_dir = save_to_dir,
                                                      save_prefix  = image_save_prefix,
                                                      seed = seed)

  mask_generator = mask_datagen.flow_from_dataframe(data_frame,
                                                    x_col = "mask",
                                                    class_mode = None,
                                                    color_mode = mask_color_mode,
                                                    target_size = target_size,
                                                    batch_size = batch_size,
                                                    save_to_dir = save_to_dir,
                                                    save_prefix  = mask_save_prefix,
                                                    seed = seed)

  df_gen = zip(image_generator, mask_generator)
    
  for (img, mask) in df_gen:
    img, mask = adjust_data(img, mask)
    yield (img,mask)

def prepare_datasets(df_train, df_val, df_test):
  
  """


  Args:


  Returns:

  """
  
  train_generator_args = dict(rotation_range=0.1,
                              width_shift_range=0.05,
                              height_shift_range=0.05,
                              shear_range=0.05,
                              zoom_range=0.05,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest')
  train_gen = batch_generator(df_train, BATCH_SIZE,
                              train_generator_args,
                              target_size=SIZE)
    
  val_gen = batch_generator(df_val, BATCH_SIZE,
                            dict(),
                            target_size=SIZE)
  
  test_gen = batch_generator(df_test, BATCH_SIZE,
                             dict(),
                             target_size=SIZE)

  return train_gen, val_gen, test_gen


def visualize_training(model):
  
  """


  Args:


  Returns:

  """
  
  plt.figure(figsize=(8,15))
  plt.subplot(3,1,1)
  plt.plot(model.history.history['loss'], 'b-', label='train_loss')
  plt.plot(model.history.history['val_loss'], 'r-', label='val_loss')
  plt.legend(loc='best')
  plt.title('Loss')

  plt.subplot(3,1,2)
  plt.plot(model.history.history['iou'], 'b-', label='train_iou')
  plt.plot(model.history.history['val_iou'], 'r-', label='val_iou')
  plt.legend(loc='best')
  plt.title('IoU')

  plt.subplot(3,1,3)
  plt.plot(model.history.history['dice_coef'], 'b-', label='train_dice_coef')
  plt.plot(model.history.history['val_dice_coef'], 'r-', label='val_dice_coef')
  plt.legend(loc='best')
  plt.title('Dice Coef')

def Visualize_predictions(model, df_test):
  
  """


  Args:


  Returns:

  """
  
  for i in range(10):
    index=np.random.randint(1,len(df_test.index))
    img = cv2.imread(df_test['image'].iloc[index])
    img = cv2.resize(img ,SIZE)
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred) > .5)
    plt.title('Prediction')
    plt.show()

