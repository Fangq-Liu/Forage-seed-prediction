import tensorflow as tf
import numpy as np
import os
import shutil
import glob
import pathlib
import re
from sklearn.model_selection import train_test_split
from model_resnet18 import resnet18 # or from model_CNN import CNN
import PIL.Image

# replace with your path
IMG_DIR_1 = 'D:/seeds_prediction/training_sample/Milk/'
IMG_DIR_2 = 'D:/seeds_prediction/training_sample/Dough/'
IMG_DIR_3 = 'D:/seeds_prediction/training_sample/Ripe/'
DEST_DIR_1 = 'D:/seeds_prediction/training_sample/Milk224/'
DEST_DIR_2 = 'D:/seeds_prediction/training_sample/Dough224/'
DEST_DIR_3 = 'D:/seeds_prediction/training_sample/Ripe224/'
CLASS_LABEL = ['Milk', 'Dough', 'Ripe']

def transform():
    IMGS = []
    def process_images(files, dest_dir):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)
        for i, f in enumerate(files):
            dest = os.path.join(dest_dir, pathlib.Path(f).name)
            PIL.Image.open(f).resize((224,224)).save(dest)
            IMGS.append(dest)
    process_images(glob.glob(f"{IMG_DIR_1}*.png"), DEST_DIR_1)
    process_images(glob.glob(f"{IMG_DIR_2}*.png"), DEST_DIR_2)
    process_images(glob.glob(f"{IMG_DIR_3}*.png"), DEST_DIR_3)
    return IMGS

def shape(IMGS):
    X, y = [], []
    for f in IMGS:
        with PIL.Image.open(f) as img:
            X.append(np.array(img.convert('RGB'))/255.0)
        parent = pathlib.Path(f).parent.name
        label = re.sub(r'[_\d]+$', '', parent)
        y.append(CLASS_LABEL.index(label))
    return np.array(X), tf.keras.utils.to_categorical(y, 3)

if __name__ == "__main__":
    # data preparation
    imgs = transform()
    X, y = shape(imgs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # model training
    model = resnet18() # or CNN()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    model.fit(X_train, y_train, epochs=30, validation_split=0.3, callbacks=[lr_scheduler])
    model.save('model_resnet18', save_format='tf') # your model name





