import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import glob
import pathlib
import re
import itertools
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import seaborn as sns 
from skimage import util
import cv2
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential

# ==================== Model Definition ====================

class BasicBlock(layers.Layer):
    """Basic block for ResNet18"""
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
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

def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = tf.keras.Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))
    return tf.keras.Sequential(layers_list, name=name)

def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # Tensor format in TensorFlow is NHWC
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
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
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)
    return model

def resnet18(im_width=224, im_height=224, num_classes=3, include_top=True):
    return _resnet(BasicBlock, [2, 2, 2, 2], im_width, im_height, num_classes, include_top)

def create_cnn_model(input_shape=(128, 128, 3)):
    """Create CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# ==================== Noise Testing Class ====================

class ComprehensiveNoiseTester:
    """Comprehensive noise tester"""
    
    def __init__(self):
        # Three specific noise types
        self.noise_config = {
            'gaussian': {'levels': [0.01, 0.05, 0.1, 0.2], 'label': 'Gaussian Noise Standard Deviation (σ)'},
            'salt_pepper': {'levels': [0.01, 0.05, 0.1, 0.2], 'label': 'Salt & Pepper Noise Ratio'}, 
            'speckle': {'levels': [0.01, 0.05, 0.1, 0.2], 'label': 'Speckle Noise Variance'}
        }
    
    def apply_noise(self, image, noise_type, level):
        """Apply specified type and intensity of noise"""
        # Ensure image is in 0-1 range
        if image.max() > 1:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image.astype(np.float32)
            
        if noise_type == 'gaussian':
            # Zero-mean Gaussian noise
            noise = np.random.normal(0, level, image_normalized.shape)
            noisy_image = image_normalized + noise
        elif noise_type == 'salt_pepper':
            noisy_image = util.random_noise(image, mode='s&p', amount=level)
        elif noise_type == 'speckle':
            # Multiplicative speckle noise
            noise = np.random.normal(0, level, image_normalized.shape)
            noisy_image = image_normalized + image_normalized * noise
        else:
            noisy_image = image_normalized.copy()
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        # Convert back if original image was 0-255 range
        if image.max() > 1:
            noisy_image = (noisy_image * 255).astype(image.dtype)
        
        return noisy_image

    
    def test_model_robustness(self, model, test_images, test_labels, model_name="Model"):
        """Comprehensive model robustness testing"""
        print(f"\n🔬 Testing {model_name} noise robustness")
        
        # Baseline performance
        baseline_pred = model.predict(test_images, verbose=0)
        y_true = np.argmax(test_labels, axis=1)
        y_pred_baseline = np.argmax(baseline_pred, axis=1)
        baseline_acc = accuracy_score(y_true, y_pred_baseline)
        
        results = {'baseline': baseline_acc}
        
        for noise_type, config in self.noise_config.items():
            print(f"  Testing {noise_type} noise...")
            accuracies = []
            
            for level in config['levels']:
                # Add noise to test set
                noisy_images = []
                for img in test_images:
                    noisy_img = self.apply_noise(img, noise_type, level)
                    noisy_images.append(noisy_img)
                
                noisy_images = np.array(noisy_images)
                
                # Ensure consistent data type
                if test_images.dtype != noisy_images.dtype:
                    noisy_images = noisy_images.astype(test_images.dtype)
                
                # Predict
                try:
                    pred = model.predict(noisy_images, verbose=0)
                    acc = accuracy_score(y_true, np.argmax(pred, axis=1))
                    accuracies.append(acc)
                except Exception as e:
                    print(f"    Level {level} prediction failed: {e}")
                    accuracies.append(0)  # Record failure
            
            results[noise_type] = accuracies
        
        return results
    
    def plot_comprehensive_results(self, results_dict, save_dir="D:/Users/Fangq/Desktop/Noise1/"):
        """Create comprehensive robustness report"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 1. Generate summary table
        summary_data = []
        detailed_data = []
        
        for model_name, results in results_dict.items():
            baseline = results['baseline']
            summary_data.append([model_name, 'Baseline', baseline, 0, 0])
            
            for noise_type, config in self.noise_config.items():
                declines = [baseline - acc for acc in results[noise_type]]
                avg_decline = np.mean(declines)
                max_decline = max(declines) if declines else 0
                summary_data.append([model_name, noise_type, avg_decline, max_decline, baseline])
                
                # Detailed data
                for i, level in enumerate(config['levels']):
                    if i < len(results[noise_type]):
                        acc = results[noise_type][i]
                        decline = baseline - acc
                        detailed_data.append([model_name, noise_type, level, acc, decline])
        
        df_summary = pd.DataFrame(summary_data, 
                                columns=['Model', 'Noise Type', 'Average Accuracy Drop', 'Max Accuracy Drop', 'Baseline Accuracy'])
        df_detailed = pd.DataFrame(detailed_data,
                                 columns=['Model', 'Noise Type', 'Noise Level', 'Accuracy', 'Accuracy Drop'])
        
        df_summary.to_csv(os.path.join(save_dir, "noise_robustness_summary.csv"), 
                         index=False, encoding='utf-8-sig')
        df_detailed.to_csv(os.path.join(save_dir, "noise_robustness_detailed.csv"), 
                          index=False, encoding='utf-8-sig')
        
        # 2. Plot detailed charts
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flat
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
        colors = colors[::-1]
        
        for idx, (noise_type, config) in enumerate(self.noise_config.items()):
            if idx < len(axes):
                ax = axes[idx]
                
                for i, (model_name, results) in enumerate(results_dict.items()):
                    baseline = results['baseline']
                    acc_values = [baseline] + results[noise_type]
                    x_values = [0] + config['levels']
                    
                    ax.plot(x_values, acc_values, marker='o', linewidth=3, 
                           label=model_name, color=colors[i], markersize=8)
                    
                    # Add value annotations
                    for j, (x, y) in enumerate(zip(x_values, acc_values)):
                        if j == 0:  # Baseline point
                            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                                      xytext=(0,15), ha='center', fontsize=10, 
                                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                        else:
                            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                                      xytext=(0,10), ha='center', fontsize=9)
                
                ax.set_xlabel(config['label'], fontsize=14)
                ax.set_ylabel('Classification Accuracy', fontsize=14)
                ax.set_title(f'Model performance under {noise_type} noise', fontsize=16, fontweight='bold')
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.3, 1.05)
        
        # Hide extra subplots
        for idx in range(len(self.noise_config), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "comprehensive_noise_analysis.pdf"), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "comprehensive_noise_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        
        # 3. Plot comparison bar chart
        plt.figure(figsize=(12, 8))
        models = list(results_dict.keys())
        noise_types = list(self.noise_config.keys())
        
        # Calculate average decline rates
        avg_declines = {}
        for model_name in models:
            declines = []
            for noise_type in noise_types:
                baseline = results_dict[model_name]['baseline']
                noise_accs = results_dict[model_name][noise_type]
                declines.extend([baseline - acc for acc in noise_accs])
            avg_declines[model_name] = np.mean(declines) if declines else 0
        
        # Plot bar chart
        x_pos = np.arange(len(models))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, noise_type in enumerate(noise_types):
            declines = []
            for model_name in models:
                baseline = results_dict[model_name]['baseline']
                noise_accs = results_dict[model_name][noise_type]
                avg_noise_decline = np.mean([baseline - acc for acc in noise_accs]) if noise_accs else 0
                declines.append(avg_noise_decline)
            
            plt.bar(x_pos + i*0.15, declines, 0.15, label=noise_type, color=colors[i % len(colors)])
        
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Average Accuracy Decrease', fontsize=14)
        plt.title('Average Performance Decrease of Each Model Under Different Noise Conditions', fontsize=16, fontweight='bold')
        plt.xticks(x_pos + 0.3, models, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "noise_comparison_bar.pdf"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive robustness analysis report saved to: {save_dir}")
        return df_summary, df_detailed

# ==================== Data Loading and Main Program ====================

def load_data_for_cnn():
    """Load data for CNN model (128x128)"""
    IMG_DIR_1 = 'D:/Code/New/FRESH/Shaixuan/FMSBP/cut/'
    IMG_DIR_2 = 'D:/Code/New/FRESH/Shaixuan/FDSBP/cut/'
    IMG_DIR_3 = 'D:/Code/New/FRESH/Shaixuan/FRSBP/cut/'
    
    CLASS_LABEL = ['Milk', 'Dough', 'Ripe']
    
    def load_and_process_images(img_dirs, target_size=(128, 128)):
        images = []
        labels = []
        
        for class_idx, img_dir in enumerate(img_dirs):
            image_paths = glob.glob(f'{img_dir}*.png')
            for image_path in image_paths:
                with PIL.Image.open(image_path) as im:
                    # Resize and convert to RGB
                    im = im.resize(target_size).convert('RGB')
                    arr = np.array(im, dtype=np.float32) / 255.0
                    images.append(arr)
                    labels.append(class_idx)
        
        return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_LABEL))
    
    img_dirs = [IMG_DIR_1, IMG_DIR_2, IMG_DIR_3]
    X, y = load_and_process_images(img_dirs, target_size=(128, 128))
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

def load_data_for_resnet():
    """Load data for ResNet18 model (224x224)"""
    IMG_DIR_1 = 'D:/Code/New/FRESH/Shaixuan/FMSBP/cut/'
    IMG_DIR_2 = 'D:/Code/New/FRESH/Shaixuan/FDSBP/cut/'
    IMG_DIR_3 = 'D:/Code/New/FRESH/Shaixuan/FRSBP/cut/'
    
    CLASS_LABEL = ['Milk', 'Dough', 'Ripe']
    
    def load_and_process_images(img_dirs, target_size=(224, 224)):
        images = []
        labels = []
        
        for class_idx, img_dir in enumerate(img_dirs):
            image_paths = glob.glob(f'{img_dir}*.png')
            for image_path in image_paths:
                with PIL.Image.open(image_path) as im:
                    # Resize and convert to RGB
                    im = im.resize(target_size).convert('RGB')
                    arr = np.array(im, dtype=np.float32) / 255.0
                    images.append(arr)
                    labels.append(class_idx)
        
        return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_LABEL))
    
    img_dirs = [IMG_DIR_1, IMG_DIR_2, IMG_DIR_3]
    X, y = load_and_process_images(img_dirs, target_size=(224, 224))
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(model, train_images, train_labels, model_name, epochs=20):
    """Train model"""
    print(f"\nTraining {model_name} model...")
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1
    )
    
    history = model.fit(train_images, train_labels, 
                       epochs=epochs, 
                       validation_split=0.3, 
                       callbacks=[lr_scheduler], 
                       verbose=1)
    
    return history

def main():
    """Main function"""
    print("Starting CNN and ResNet18 model noise robustness comparison experiment")
    print("=" * 60)
    
    # Create save directory
    save_dir = "D:/Users/Fangq/Desktop/Noise3/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Load data
    print("1. Loading data...")
    cnn_train_images, cnn_test_images, cnn_train_labels, cnn_test_labels = load_data_for_cnn()
    resnet_train_images, resnet_test_images, resnet_train_labels, resnet_test_labels = load_data_for_resnet()
    
    print(f"CNN data - Training set: {cnn_train_images.shape}, Test set: {cnn_test_images.shape}")
    print(f"ResNet data - Training set: {resnet_train_images.shape}, Test set: {resnet_test_images.shape}")
    
    # 2. Create and train models
    print("\n2. Creating and training models...")
    
    # CNN model
    cnn_model = create_cnn_model(input_shape=cnn_train_images.shape[1:])
    cnn_history = train_model(cnn_model, cnn_train_images, cnn_train_labels, "CNN", epochs=30)
    
    # ResNet18 model
    resnet_model = resnet18(im_width=224, im_height=224, num_classes=3)
    resnet_history = train_model(resnet_model, resnet_train_images, resnet_train_labels, "ResNet18", epochs=30)
    
    # 3. Baseline performance evaluation
    print("\n3. Baseline performance evaluation...")
    
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(cnn_test_images, cnn_test_labels, verbose=0)
    resnet_test_loss, resnet_test_acc = resnet_model.evaluate(resnet_test_images, resnet_test_labels, verbose=0)
    
    print(f"CNN model baseline accuracy: {cnn_test_acc:.4f}")
    print(f"ResNet18 model baseline accuracy: {resnet_test_acc:.4f}")
    
    # 4. Noise robustness testing
    print("\n4. Noise robustness testing...")
    
    noise_tester = ComprehensiveNoiseTester()
    
    # Test CNN model
    cnn_results = noise_tester.test_model_robustness(cnn_model, cnn_test_images, cnn_test_labels, "CNN")
    
    # Test ResNet18 model
    resnet_results = noise_tester.test_model_robustness(resnet_model, resnet_test_images, resnet_test_labels, "ResNet18")
    
    results_dict = {
        "CNN": cnn_results,
        "ResNet18": resnet_results
    }
    
    # 5. Generate comprehensive report
    print("\n5. Generating comprehensive report...")
    summary_df, detailed_df = noise_tester.plot_comprehensive_results(results_dict, save_dir)
    
    # 6. Print key conclusions
    print("\n" + "=" * 60)
    print("Key Conclusions Summary")
    print("=" * 60)
    
    for model_name, results in results_dict.items():
        baseline = results['baseline']
        print(f"\n{model_name} model:")
        print(f"  Baseline accuracy: {baseline:.4f}")
        
        # Calculate overall robustness score (lower decrease is better)
        total_decline = 0
        noise_count = 0
        
        for noise_type in noise_tester.noise_config.keys():
            declines = [baseline - acc for acc in results[noise_type]]
            avg_decline = np.mean(declines) if declines else 0
            max_decline = max(declines) if declines else 0
            total_decline += avg_decline
            noise_count += 1
            
            print(f"  {noise_type}: Average decrease {avg_decline:.4f}, Max decrease {max_decline:.4f}")
        
        robustness_score = 1 - (total_decline / noise_count)  # Robustness score (higher is better)
        print(f"  Overall robustness score: {robustness_score:.4f}")

if __name__ == '__main__':
    # Ensure TensorFlow uses GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU for training")
    else:
        print("Using CPU for training")
    
    main()

