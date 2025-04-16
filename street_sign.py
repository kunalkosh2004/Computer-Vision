import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from my_utils import split_data, order_test_set, create_generators
from deeplearning_models import street_model
import tensorflow
            

if __name__ == "__main__":
    # if False:
    #     path_to_data = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/Train"
    #     path_to_save_train = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/training_data/train"
    #     path_to_save_val = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/training_data/val"
    #     split_data(path_to_data,path_to_save_train,path_to_save_val)
        
    # path_to_images = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/Test"
    # path_to_csv = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/Test.csv"
    # order_test_set(path_to_images,path_to_csv)
    
    path_to_train = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/training_data/train"
    path_to_val = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/training_data/val"
    path_to_test = "/Users/kunalkoshta/Desktop/Computer-Vision/Computer-Vision/archive/Test"
    batch_size = 64
    epochs=15
    lr = 0.0001
    
    train_generator, val_generator, test_generator = create_generators(batch_size,path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes
    
    TRAIN = False
    TEST = True
    
    if TRAIN:
        path_to_save_model = "./Models/best_model.keras"
        os.makedirs("./Models", exist_ok=True)
        ckpt_server = ModelCheckpoint(
            path_to_save_model,
            monitor = 'val_accuracy',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose=1
        )
        
        early_stop = EarlyStopping(monitor = 'val_accuracy', patience=10)
        
        model = street_model(nbr_classes)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(
            train_generator,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = val_generator,
            callbacks = [ckpt_server, early_stop]
        )
        
    if TEST:
        model = tensorflow.keras.models.load_model('./Models/best_model.keras')
        model.summary()
        
        print("Evaluating Validation Set: ")
        model.evaluate(val_generator)
        
        print("Evaluating Test Set: ")
        model.evaluate(test_generator)