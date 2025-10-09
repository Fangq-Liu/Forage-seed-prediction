import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import glob
import pathlib
import re
import itertools
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns 

# Image directories - replace with your actual paths
IMG_DIR_1 = 'your_path/Milk/'
IMG_DIR_2 = 'your_path/Dough/'
IMG_DIR_3 = 'your_path/Ripe/'  

DEST_DIR_1 = 'your_path/Milk128/'
DEST_DIR_2 = 'your_path/Dough128/'
DEST_DIR_3 = 'your_path/Ripe128/' 

IMGS = []
CLASS_LABEL = ['Milk', 'Dough', 'Ripe']  # Class labels

# Output directory for results
OUTPUT_DIR = 'your_output_path/'

class BasicBlock(layers.Layer):
    """Basic ResNet block implementation"""
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
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
    """Create a ResNet layer with specified blocks"""
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
    # TensorFlow tensor channel order is NHWC
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
        # Add Dropout layer for uncertainty analysis
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model

def resnet18(im_width=224, im_height=224, num_classes=3, include_top=True):
    """Create ResNet-18 model"""
    return _resnet(BasicBlock, [2, 2, 2, 2], im_width, im_height, num_classes, include_top)

def create_uncertainty_heatmap(y_true, y_pred, uncertainty_values, class_names, figsize=(12, 10)):
    """Create comprehensive uncertainty heatmap visualization"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate average uncertainty for each confusion matrix cell
    n_classes = cm.shape[0]
    uncertainty_matrix = np.zeros((n_classes, n_classes))
    count_matrix = np.zeros((n_classes, n_classes))
    
    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            # Find indices where true class is true_class and predicted class is pred_class
            indices = np.where((y_true == true_class) & (y_pred == pred_class))[0]
            if len(indices) > 0:
                uncertainty_matrix[true_class, pred_class] = np.mean(uncertainty_values[indices])
                count_matrix[true_class, pred_class] = len(indices)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Standard confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. Uncertainty heatmap
    sns.heatmap(uncertainty_matrix, annot=True, fmt='.3f', cmap='Reds',
               xticklabels=class_names, yticklabels=class_names,
               ax=axes[0, 1])
    axes[0, 1].set_title('Average Prediction Uncertainty', fontsize=14)
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    
    # 3. Sample count heatmap
    sns.heatmap(count_matrix, annot=True, fmt='.0f', cmap='Greens',
               xticklabels=class_names, yticklabels=class_names,
               ax=axes[1, 0])
    axes[1, 0].set_title('Sample Count per Cell', fontsize=14)
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    
    # 4. Uncertainty distribution histogram
    axes[1, 1].hist(uncertainty_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(uncertainty_values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(uncertainty_values):.3f}')
    axes[1, 1].set_xlabel('Uncertainty Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Uncertainty Distribution', fontsize=14)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}FSBP_uncertainty_heatmap.pdf", format="pdf")
    plt.show()
    
    return uncertainty_matrix

def plot_high_uncertainty_samples(test_images, y_true, y_pred, uncertainty_values, 
                                 mc_predictions, mean_predictions, class_names, n_samples=5):
    """Visualize samples with highest uncertainty"""
    # Find samples with highest uncertainty
    high_unc_indices = np.argsort(uncertainty_values)[-n_samples:]
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(high_unc_indices):
        # Display original image
        axes[i, 0].imshow(test_images[idx])
        axes[i, 0].set_title(f'Sample {idx}\nTrue: {class_names[y_true[idx]]}')
        axes[i, 0].axis('off')
        
        # Display prediction probability distribution
        pred_class = y_pred[idx]
        axes[i, 1].bar(range(len(class_names)), mean_predictions[idx])
        axes[i, 1].set_xticks(range(len(class_names)))
        axes[i, 1].set_xticklabels(class_names, rotation=45)
        axes[i, 1].set_title(f'Pred: {class_names[pred_class]}\nUncertainty: {uncertainty_values[idx]:.3f}')
        axes[i, 1].set_ylabel('Probability')
        
        # Display MC Dropout prediction variability
        mc_probs = mc_predictions[:, idx, pred_class]
        axes[i, 2].hist(mc_probs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[i, 2].axvline(np.mean(mc_probs), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(mc_probs):.3f}')
        axes[i, 2].set_xlabel('Predicted Probability')
        axes[i, 2].set_ylabel('Frequency')
        axes[i, 2].set_title('MC Dropout Distribution')
        axes[i, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}FSBP_high_uncertainty_samples.pdf", format="pdf")
    plt.show()

def generate_uncertainty_report(y_true, y_pred, uncertainty_values, class_names):
    """Generate comprehensive uncertainty analysis report"""
    print("\n" + "="*50)
    print("Uncertainty Analysis Report")
    print("="*50)
    
    print(f"Average Uncertainty: {np.mean(uncertainty_values):.4f} ± {np.std(uncertainty_values):.4f}")
    print(f"Uncertainty Range: [{np.min(uncertainty_values):.4f}, {np.max(uncertainty_values):.4f}]")
    
    # Analyze uncertainty by true class
    print("\nUncertainty Analysis by True Class:")
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            class_uncertainty = uncertainty_values[class_indices]
            print(f"  {class_name}: {np.mean(class_uncertainty):.4f} ± {np.std(class_uncertainty):.4f} "
                  f"(n={len(class_indices)})")
    
    # Analyze uncertainty for correct vs incorrect classifications
    correct_mask = (y_true == y_pred)
    print(f"\nAverage Uncertainty for Correct Classifications: {np.mean(uncertainty_values[correct_mask]):.4f}")
    print(f"Average Uncertainty for Incorrect Classifications: {np.mean(uncertainty_values[~correct_mask]):.4f}")
    
    # Calculate correlation between uncertainty and misclassification
    error_correlation = np.corrcoef(uncertainty_values, (y_true != y_pred).astype(int))[0, 1]
    print(f"Correlation between Uncertainty and Misclassification: {error_correlation:.4f}")
    
    # Save report to file
    report_data = {
        'mean_uncertainty': np.mean(uncertainty_values),
        'std_uncertainty': np.std(uncertainty_values),
        'min_uncertainty': np.min(uncertainty_values),
        'max_uncertainty': np.max(uncertainty_values),
        'uncertainty_correlation_with_error': error_correlation
    }
    
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            class_uncertainty = uncertainty_values[class_indices]
            report_data[f'{class_name}_mean_uncertainty'] = np.mean(class_uncertainty)
            report_data[f'{class_name}_std_uncertainty'] = np.std(class_uncertainty)
    
    report_df = pd.DataFrame([report_data])
    report_df.to_csv(f"{OUTPUT_DIR}FSBP_uncertainty_report.csv", index=False)
    print(f"\nUncertainty report saved to: {OUTPUT_DIR}FSBP_uncertainty_report.csv")

def transform():
    """Transform and resize images for training"""
    global IMGS
    fs1 = glob.glob('{}*.png'.format(IMG_DIR_1))
    fs2 = glob.glob('{}*.png'.format(IMG_DIR_2))
    fs3 = glob.glob('{}*.png'.format(IMG_DIR_3))  # Additional intermediate state images

    def process_images(image_paths, dest_dir):
        """Process and resize individual images"""
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processed {i} images")
            with PIL.Image.open(image_path) as im:
                dest = os.path.join(dest_dir, pathlib.Path(image_path).name)
                im.resize((224, 224)).save(dest)
                IMGS.append(dest)

    # Create destination directories
    for dest_dir in [DEST_DIR_1, DEST_DIR_2, DEST_DIR_3]:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)

    process_images(fs1, DEST_DIR_1)
    process_images(fs2, DEST_DIR_2)
    process_images(fs3, DEST_DIR_3)  # Process new category images

def shape():
    """Prepare dataset by loading and labeling images"""
    M = []
    y_values = []
    for fs in IMGS:
        with PIL.Image.open(fs) as im:
            arr = np.array(im.convert('RGB'), dtype=np.float32) / 255.0
            M.append(arr)
            
            # Extract directory name and remove possible numeric suffixes
            directory_name = pathlib.Path(fs).parent.name
            # Assume directory name format is "ClassName_digits"
            # Use regex to remove trailing digits and underscores
            label_name = re.sub(r'[_\d]+$', '', directory_name)  # Remove trailing digits and underscores
            
            # Determine class label
            try:
                class_index = CLASS_LABEL.index(label_name)
                y_values.append(class_index)
            except ValueError as e:
                raise ValueError(f"Label name {label_name} not found in CLASS_LABEL") from e

    X = np.array(M)
    y = tf.keras.utils.to_categorical(y_values, num_classes=len(CLASS_LABEL))
    return X, y

def main():
    """Main execution function"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Transform images and prepare dataset
    transform()
    X, y = shape()
    train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    # Define and compile model
    model = resnet18()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        mode='auto', 
        min_delta=0.0001
    )
    
    # Train model
    history = model.fit(train_images, train_labels, epochs=30, 
                        validation_split=0.3, callbacks=[lr_scheduler])
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    
    # Get predictions
    predictions = model.predict(test_images)
    print(predictions)
    
    # Calculate evaluation metrics
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Precision, Recall, F1-Score
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Calculate AUC
    try:
        auc = roc_auc_score(test_labels, predictions, multi_class='ovr')
    except ValueError as e:
        print("AUC calculation failed:", e)
        auc = None
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Store all metrics in a list
    metrics_list = []
    metrics_list.append(('Accuracy', accuracy))
    metrics_list.append(('Precision', precision))
    metrics_list.append(('Recall', recall))
    metrics_list.append(('F1-Score', f1))
    if auc is not None:
        metrics_list.append(('AUC', auc))
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    for metric_name, metric_value in metrics_list:
        print(f"{metric_name}: {metric_value:.4f}")
    
    # ==================== Uncertainty Analysis ====================
    print("\nStarting uncertainty analysis...")
    
    # Perform MC Dropout predictions
    n_mc_samples = 100  # Number of Monte Carlo samples
    mc_predictions = []
    
    # Enable Dropout layers during prediction
    for i in range(n_mc_samples):
        if i % 20 == 0:
            print(f"MC Dropout sampling progress: {i}/{n_mc_samples}")
        # Use training=True to enable Dropout
        mc_pred = model(test_images, training=True)
        mc_predictions.append(mc_pred.numpy())
    
    mc_predictions = np.array(mc_predictions)  # shape: (n_mc_samples, n_test, n_classes)
    
    # Calculate uncertainty metrics
    mean_predictions = np.mean(mc_predictions, axis=0)
    uncertainty = np.std(mc_predictions, axis=0)
    
    # Epistemic uncertainty (standard deviation of predicted class)
    epistemic_uncertainty = uncertainty[np.arange(len(uncertainty)), y_pred]
    
    # Create uncertainty heatmap
    create_uncertainty_heatmap(y_true, y_pred, epistemic_uncertainty, CLASS_LABEL)
    
    # Visualize high uncertainty samples
    plot_high_uncertainty_samples(test_images, y_true, y_pred, epistemic_uncertainty, 
                                 mc_predictions, mean_predictions, CLASS_LABEL)
    
    # Generate uncertainty report
    generate_uncertainty_report(y_true, y_pred, epistemic_uncertainty, CLASS_LABEL)
    
    # ==================== Original Code Continues ====================
    
    # Calculate ROC curve data (original code)
    try:
        fpr, tpr, thresholds = roc_curve(test_labels.ravel(), predictions.ravel())
    except Exception as e:
        print("ROC curve calculation error:", e)
        fpr, tpr, thresholds = [], [], []
    
    # Save ROC data (original code)
    if len(fpr) > 0:
        roc_data = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr,
            'Thresholds': thresholds
        })
        roc_data.to_csv(f"{OUTPUT_DIR}FSBP_ROC.csv", index=False)
        
        # Plot ROC curve (original code)
        plt.figure()
        plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"{OUTPUT_DIR}FSBP_ROC.pdf", format="pdf")
        plt.show()

    # Plot confusion matrix
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('FSBP Confusion Matrix')
    plt.colorbar()

    # Set class labels as axis labels for confusion matrix
    plt.xticks(range(len(CLASS_LABEL)), CLASS_LABEL)
    plt.yticks(range(len(CLASS_LABEL)), CLASS_LABEL)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display values in confusion matrix
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.savefig(f"{OUTPUT_DIR}FSBP_confusion_matrix.pdf", format="pdf")
    
    # Visualize training process
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}FSBP_training_history.pdf", format="pdf")
    plt.show()
    
    # Visualization of training samples
    plt.rcParams['font.sans-serif'] = ['Arial']
    # The first 30 images used for training are shown following the first image
    a, b = 5, 6
    plt.figure(figsize=(10, 10))
    plt.suptitle('The first {} images used for training'.format(a * b))
    for i in range(a * b):
        plt.subplot(a, b, i + 1)
        plt.xticks([])  
        plt.yticks([])  
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # Ensure using integer index to access CLASS_LABEL
        plt.xlabel(CLASS_LABEL[np.argmax(train_labels[i])])
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig(f"{OUTPUT_DIR}FSBP_training_samples.pdf", format="pdf")
    plt.show()

    # All the test images are output finally, with the prediction for each image
    n = len(test_images)
    col = 6
    row = n // col if n % col == 0 else n // col + 1
    plt.figure(figsize=(10, 15)) 
    plt.suptitle('All {} predicted results'.format(n))
    
    # Assume test_labels is the actual class index array for test images
    # and predictions is the predicted probability array from the model
    for i in range(n):
        plt.subplot(row, col, i + 1)
        plt.xticks([])  
        plt.yticks([])  
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        prediction = predictions[i]
        res = np.argmax(prediction)  # Get the class index with highest prediction probability
        actual_label_index = np.argmax(test_labels[i])  # Get the class index with highest probability in actual label (if one-hot encoded)
        if res == actual_label_index:
            plt.xlabel('{} {:.1f}%'.format(CLASS_LABEL[res], prediction[res] * 100), color='green')
        else:
            plt.xlabel('{} {:.1f}%'.format(CLASS_LABEL[res], prediction[res] * 100), color='red')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"{OUTPUT_DIR}FSBP_test_predictions.pdf", format="pdf")
    plt.show()

if __name__ == '__main__':
    main()

