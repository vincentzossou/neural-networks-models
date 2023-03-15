from __future__ import print_function

import os, sys, pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from loss_function import *
from train_generator import *
from val_generator import *
from unet import *


tf.config.list_physical_devices('GPU')

metric = SemanticLossFunction()
model_loaded = AttentionResUnetModel()
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def train():

    train_dataset = TrainGenerator(train_images_number)
    val_dataset = ValGenerator(val_images_number)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    """ Hyperparameters """
    input_shape = (512, 512, 4)
    lr = 1e-3
    _epochs = 50
    _loss=metric.jacard_coef_loss
    _metrics=[metric.jacard_coef, metric.dice_coef, metric.sensitivity, metric.specificity]

    model = model_loaded.build_unet(input_shape)
    model.compile(optimizer=Adam(learning_rate=lr), loss=_loss, metrics=_metrics)

    # Callbacks
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir="./logs")

    history = model.fit(train_dataset, epochs=_epochs, verbose=1,
                        validation_data=val_dataset,
                        callbacks=[model_checkpoint, tensorboard_callback])

    file = open('trainHistDict512.txt', 'wb') #training history
    pickle.dump(history.history, file)

    return history

if __name__ == '__main__':
    history = train()
    