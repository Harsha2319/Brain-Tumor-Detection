import tensorflow as tf

from utils import extract_datafile
from utils import creating_dataframe
from utils import split_dataset
from utils import prepare_datasets
from utils import visualize_training
from utils import Visualize_predictions 

from model import unet

from metrics import dice_coef
from metrics import dice_coef_loss
from metrics import bce_dice_loss
from metrics import iou

dataset_path = "../data/Brain_MRI_Data.zip"
dir_extracted = "../data/lgg-mri-segmentation/"
extract_to = "../data"
model_path = "../model/BrainTumorDetectionModel.h5"

# Training Parameters
SIZE = (256, 256)
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

extract_datafile(dataset_path, dir_extracted, extract_to)
df = creating_dataframe()
df_train, df_val, df_test = split_dataset(df)
train_gen, val_gen, test_gen = prepare_datasets(df_train, df_val, df_test)

model = unet(input_size=(SIZE[0], SIZE[1], 3))

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

callbacks = [tf.keras.callbacks.ModelCheckpoint('brainMRI_Segment.hdf5', verbose=0, save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-11),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)]

model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[iou, dice_coef])

history = model.fit(train_gen,
                    steps_per_epoch=len(df_train) // BATCH_SIZE, 
                    epochs=EPOCHS, 
                    callbacks=callbacks,
                    validation_data = val_gen,
                    validation_steps=len(df_val) // BATCH_SIZE)

visualize_training(model)

results = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)
print("Test IOU: ",results[1])
print("Test Dice Coefficent: ",results[2])

Visualize_predictions(model, df_test)

model.save(model_path)

