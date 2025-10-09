import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, GlobalAveragePooling2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import random
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import shutil

# Dataset root directory - replace with your actual path
root = "your_dataset_path/19Bands/"

EPOCHS = 50
BATCH_SIZE = 8
LR = 0.001  # Increased learning rate
IMG_SIZE = (128, 128)  # Reduced resolution for faster training
NUM_BANDS = 19
NUM_CLASSES = 3

np.random.seed(42)
tf.random.set_seed(42)

class BasicBlock(layers.Layer):
    """Basic ResNet block implementation"""
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.out_channel = out_channel
        self.strides = strides
        
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x
    
    def get_config(self):
        config = super(BasicBlock, self).get_config()
        config.update({
            'out_channel': self.out_channel,
            'strides': self.strides,
        })
        return config

def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    """Create a ResNet layer with specified number of blocks"""
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)

def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    """Build ResNet architecture"""
    input_image = layers.Input(shape=(im_height, im_width, NUM_BANDS), dtype="float32")  # Modified for 19 channels
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)
    return model

def resnet18(im_width=128, im_height=128, num_classes=3, include_top=True):
    """Create ResNet-18 model for multi-band image classification"""
    return _resnet(BasicBlock, [2, 2, 2, 2], im_width, im_height, num_classes, include_top)

class MultiBandDataset(tf.keras.utils.Sequence):
    """Fixed multi-band dataset class for handling 19-band images"""
    
    def __init__(self, base_images, labels, batch_size, num_bands, img_size):
        self.base_images = base_images
        self.labels = labels
        self.batch_size = batch_size
        self.num_bands = num_bands
        self.img_size = img_size
        self.indices = np.arange(len(base_images))
        
    def __len__(self):
        return int(np.ceil(len(self.base_images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch data - 19 bands
        batch_images = np.zeros((len(batch_indices), *self.img_size, self.num_bands), dtype=np.float32)
        batch_labels = np.zeros((len(batch_indices), NUM_CLASSES), dtype=np.float32)
        
        for i, batch_idx in enumerate(batch_indices):
            base_path = self.base_images[batch_idx]
            base_dir = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            
            # Load all 19 band images
            for band in range(self.num_bands):
                if band == 0:
                    # band1: file without parentheses
                    img_name = base_name
                else:
                    # band2-band19: use (1) to (18)
                    img_name = base_name.replace('.png', f'({band}).png')
                
                img_path = os.path.join(base_dir, img_name)
                
                try:
                    # Load image in grayscale mode
                    img = load_img(img_path, target_size=self.img_size, color_mode='grayscale')
                    img_array = img_to_array(img) / 255.0
                    batch_images[i, :, :, band] = img_array[:, :, 0]  # Take single channel
                except FileNotFoundError:
                    print(f"Error: Missing image {img_path}")
                    # Use random noise if image not found (for debugging)
                    batch_images[i, :, :, band] = np.random.rand(*self.img_size)
            
            batch_labels[i] = self.labels[batch_idx]
        
        print(f"Batch {idx}: Loaded {len(batch_indices)} samples, shape {batch_images.shape}")
        return batch_images, batch_labels

def read_and_split_data(root: str, num_bands: int, test_size: float = 0.2):
    """Fixed data reading and splitting function"""
    
    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"Found classes: {classes}")
    print(f"Class indices: {class_indices}")
    
    # Collect all base image paths and labels
    base_images = []
    labels = []
    
    for class_name in classes:
        class_path = os.path.join(root, class_name)
        class_idx = class_indices[class_name]
        
        # Get all base image files (files without parentheses)
        base_files = []
        for f in os.listdir(class_path):
            if f.endswith('.png') and not re.search(r'\(\d+\)\.png$', f):
                base_files.append(f)
        
        print(f"Class {class_name}: found {len(base_files)} base images")
        
        for base_file in base_files:
            base_path = os.path.join(class_path, base_file)
            
            # Verify all 19 band files exist
            all_bands_exist = True
            for band in range(num_bands):
                if band == 0:
                    band_file = base_file
                else:
                    band_file = base_file.replace('.png', f'({band}).png')
                
                band_path = os.path.join(class_path, band_file)
                if not os.path.exists(band_path):
                    print(f"Warning: Missing band {band} for {base_file}")
                    all_bands_exist = False
                    break
            
            if all_bands_exist:
                base_images.append(base_path)
                labels.append(class_idx)
            else:
                print(f"Skipping {base_file} due to missing bands")
    
    print(f"Total valid base images: {len(base_images)}")
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split training and test sets
    train_base, test_base, train_labels, test_labels = train_test_split(
        base_images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Convert to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=NUM_CLASSES)
    test_labels = to_categorical(test_labels, num_classes=NUM_CLASSES)
    
    print(f"Training samples: {len(train_base)}")
    print(f"Test samples: {len(test_base)}")
    
    return train_base, test_base, train_labels, test_labels

def plot_training_history(history):
    """Plot training history (accuracy and loss)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def debug_data_loading(train_base, test_base, num_bands, img_size):
    """Debug data loading to verify file paths and image loading"""
    print("\n=== Debug Data Loading ===")
    
    # Check all bands for first training sample
    sample_path = train_base[0]
    sample_dir = os.path.dirname(sample_path)
    sample_name = os.path.basename(sample_path)
    
    print(f"Sample path: {sample_path}")
    print(f"Sample directory: {sample_dir}")
    print(f"Sample name: {sample_name}")
    
    for band in range(min(3, num_bands)):  # Only check first 3 bands
        if band == 0:
            band_file = sample_name
        else:
            band_file = sample_name.replace('.png', f'({band}).png')
        
        band_path = os.path.join(sample_dir, band_file)
        print(f"Band {band}: {band_path} - Exists: {os.path.exists(band_path)}")
        
        if os.path.exists(band_path):
            img = load_img(band_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img)
            print(f"  Shape: {img_array.shape}, Min: {img_array.min()}, Max: {img_array.max()}")

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    """Custom model checkpoint callback to avoid HDF5 issues"""
    def __init__(self, filepath, monitor='val_accuracy', save_best_only=True):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = -np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if current > self.best:
            print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}')
            self.best = current
            if self.save_best_only:
                # Save using SavedModel format
                model_path = os.path.join(self.filepath, f"epoch_{epoch+1}")
                print(f'Saving model to {model_path}')
                self.model.save(model_path, save_format='tf')
        else:
            print(f'\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}')

def main():
    # Set up device
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU for training")
    else:
        print("Using CPU for training")
    
    # Read and split data
    print("Loading data...")
    train_base, test_base, train_labels, test_labels = read_and_split_data(
        root, num_bands=NUM_BANDS, test_size=0.2
    )
    
    # Debug data loading
    debug_data_loading(train_base, test_base, NUM_BANDS, IMG_SIZE)
    
    # Create datasets
    train_dataset = MultiBandDataset(
        train_base, train_labels, BATCH_SIZE, NUM_BANDS, IMG_SIZE
    )
    test_dataset = MultiBandDataset(
        test_base, test_labels, BATCH_SIZE, NUM_BANDS, IMG_SIZE
    )
    
    # Test one batch to ensure correct data loading
    print("\n=== Testing Data Loading ===")
    test_batch_images, test_batch_labels = train_dataset[0]
    print(f"Batch images shape: {test_batch_images.shape}")
    print(f"Batch labels shape: {test_batch_labels.shape}")
    print(f"Batch labels: {np.argmax(test_batch_labels, axis=1)}")
    
    # Create ResNet18 model
    print("Creating ResNet18 model...")
    model = resnet18(im_width=IMG_SIZE[0], im_height=IMG_SIZE[1], num_classes=NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Create model save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'best_resnet18_model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up callbacks - using custom callback to avoid HDF5 issues
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20, restore_best_weights=True, min_delta=0.01
        ),
        CustomModelCheckpoint(model_dir, monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    print("Final evaluation:")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predict and display classification report
    y_pred = model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_labels, axis=1)[:len(y_pred_classes)]
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Dough', 'Milk', 'Ripe']))
    
    # Add confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(['Dough', 'Milk', 'Ripe']))
    plt.xticks(tick_marks, ['Dough', 'Milk', 'Ripe'], rotation=45)
    plt.yticks(tick_marks, ['Dough', 'Milk', 'Ripe'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add numbers to the plot
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

