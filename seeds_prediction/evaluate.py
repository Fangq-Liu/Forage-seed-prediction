import numpy as np
import tensorflow as tf
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from train_resnet18 import transform, shape, CLASS_LABEL
from sklearn.model_selection import train_test_split
from model_resnet18 import BasicBlock

# model loading
model = tf.keras.models.load_model('model_resnet18', custom_objects={'BasicBlock': BasicBlock})
# model = tf.keras.models.load_model('model_CNN')

# data loading
imgs = transform()
X, y = shape(imgs)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# evaluate
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_cls = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred_cls))
print("Precision:", precision_score(y_true, y_pred_cls, average='macro'))
print("Recall:", recall_score(y_true, y_pred_cls, average='macro'))
print("F1:", f1_score(y_true, y_pred_cls, average='macro'))
print("AUC:", roc_auc_score(y_test, y_pred, multi_class='ovr'))

# confusion matrix
cm = confusion_matrix(y_true, y_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_LABEL, yticklabels=CLASS_LABEL)
plt.show()
#plt.savefig("confusion_matrix.pdf")

# ROC curve
try:
    auc_value = roc_auc_score(y_test, y_pred, multi_class='ovr') 
except Exception as e:
    print("AUC计算失败:", e)
    auc_value = 0.5  
fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel()) 
plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
plt.show()
#plt.savefig("roc_curve.pdf")

# test_smaple
def visualize_predictions(X_test, y_test, predictions, class_labels):
    plt.rcParams['font.sans-serif'] = ['Arial']
    n = len(X_test)
    col = 6
    row = n // col + (1 if n % col else 0)
    plt.figure(figsize=(10, 15))
    plt.suptitle(f'All {n} predicted results')
    for i in range(n):
        plt.subplot(row, col, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test[i], cmap=plt.cm.binary)
        
        pred_label = np.argmax(predictions[i])
        true_label = np.argmax(y_test[i])
        confidence = np.max(predictions[i]) * 100
        
        color = 'green' if pred_label == true_label else 'red'
        plt.xlabel(f'{class_labels[pred_label]} {confidence:.1f}%', color=color)
    
    plt.subplots_adjust(top=0.97, hspace=0.5)
    #plt.savefig("test_smaple.pdf")
    plt.show()

visualize_predictions(X_test, y_test, y_pred, CLASS_LABEL)



