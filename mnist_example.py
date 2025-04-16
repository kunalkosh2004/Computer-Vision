import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Dense, MaxPooling2D
from tensorflow.python.keras import activations
from deeplearning_models import functional_model, MyCustomModel

seq_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        
        Conv2D(128, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

if __name__ == "__main__":
    (X_train,y_train),(X_test,y_test) = tensorflow.keras.datasets.mnist.load_data()
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    y_train = tensorflow.keras.utils.to_categorical(y_train,10)
    y_test = tensorflow.keras.utils.to_categorical(y_test,10)
    
    model = MyCustomModel()
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train, batch_size = 32, epochs=5, validation_split=0.2)
    model.evaluate(X_test,y_test, batch_size=32)