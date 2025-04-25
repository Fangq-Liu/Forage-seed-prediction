import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from model_resnet18 import resnet18, BasicBlock  # Import custom layers from model file

class ImagePredictor:
    def __init__(self, model_path='model_resnet18'):
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'BasicBlock': BasicBlock}
            )
            self.class_labels = ['Milk', 'Dough', 'Ripe']  # Must match training labels
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise

    def preprocess_image(self, image_path, target_size=(224, 224)):
        try:
            # Open and convert image
            img = Image.open(image_path).convert('RGB')
            
            # Resize and normalize
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"Image preprocessing failed [{image_path}]: {str(e)}")
            return None

    def predict_image(self, image_path, show_result=False):
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None, None

        try:
            # Perform prediction
            predictions = self.model.predict(processed_img)
            pred_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Visualize results
            if show_result:
                self._visualize_prediction(image_path, pred_class, confidence)
            
            return self.class_labels[pred_class], float(confidence)
        except Exception as e:
            print(f"Prediction failed [{image_path}]: {str(e)}")
            return None, None

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            label, conf = self.predict_image(path)
            if label is not None:
                results.append((path, label, conf))
        return results

    def _visualize_prediction(self, image_path, label, confidence):
        plt.figure(figsize=(8, 8))
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Prediction: {label}\nConfidence: {confidence:.1%}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Initialize predictor
    predictor = ImagePredictor()
    
    # Handle command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image prediction: python predict.py <image_path> [--show]")
        print("  Batch prediction: python predict.py <directory_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    show_image = '--show' in sys.argv

    # Determine input type
    if Path(input_path).is_dir():
        # Batch prediction mode
        image_files = list(Path(input_path).glob("*.[pj][np]g"))  # Support png/jpg
        print(f"Found {len(image_files)} images")
        results = predictor.predict_batch([str(f) for f in image_files])
        
        # Print results table
        print("\n{:<30} | {:<10} | {}".format("Image Path", "Label", "Confidence"))
        print("-" * 60)
        for path, label, conf in results:
            print(f"{Path(path).name:<30} | {label:<10} | {conf:.1%}")
    else:
        # Single prediction mode
        label, confidence = predictor.predict_image(input_path, show_result=show_image)
        if label:
            print(f"\nPrediction result: {label}")
            print(f"Confidence: {confidence:.1%}")
