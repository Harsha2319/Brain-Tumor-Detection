import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
  
  """


  Args:


  Returns:

  """
  
  smooth=1
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  intersection = K.sum(y_true * y_pred)
  union = K.sum(y_true) + K.sum(y_pred)
  return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
  
  """


  Args:


  Returns:

  """
  
  return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
  
  """


  Args:


  Returns:

  """
  
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

def iou(y_true, y_pred):
  
  """


  Args:


  Returns:

  """
  
  smooth=1
  intersection = K.sum(y_true * y_pred)
  sum_ = K.sum(y_true + y_pred)
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return jac

