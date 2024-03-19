# import libraries
import warnings
warnings.filterwarnings('ignore')

# ML libraries 
import tensorflow as tf
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# set seeds to get standard results
import keras
keras.utils.set_random_seed(44)
tf.config.experimental.enable_op_determinism()


class Eval:
    def __init__(self, RetinaDatasetPreprocessor, Model) -> None:
        self.train_images = RetinaDatasetPreprocessor.train
        self.train_masks = RetinaDatasetPreprocessor.train_mask
        self.test_images = RetinaDatasetPreprocessor.test
        self.test_masks = RetinaDatasetPreprocessor.test_mask
        self.model = Model.model
        self.history = Model.history

    def data_size(self):
        # printso out the number of images in each set and masked sets
        print("Train set size:", len(self.train_images))
        print("Train masks set size:", len(self.train_masks))
        print("Test set size:", len(self.test_images))
        print("Test set size:", len(self.test_masks))

    def evaluate_model(self):
        # passes the evaluation from the model's evaluation
        train_loss, train_accuracy, train_iou, train_precision, train_recall = self.model.evaluate(self.train_images, self.train_masks)
        test_loss, test_accuracy, test_iou, test_precision, test_recall = self.model.evaluate(self.test_images, self.test_masks)

        # Print the evaluation results
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Train IoU: {train_iou}")
        print(f"Train Precision: {train_precision}")
        print(f"Train Recall: {train_recall}")
        print(f"\nTest Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test IoU: {test_iou}")
        print(f"Test Precision: {test_precision}")
        print(f"Test Recall: {test_recall}")

    def generate_metrics_table(self):
        # Calculates and displays a table of metrics 
        y_true = []
        y_pred = []
        
        accuracy_metric = tf.keras.metrics.Accuracy()
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()
        iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
        
        for image, mask in zip(self.test_images, self.test_masks):
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))[0].squeeze()
            pred_mask_binary = np.round(pred_mask).astype('int32')
            true_mask_binary = np.round(mask).astype('int32').squeeze()
            
            # Flatten the masks for metric calculations
            y_true.extend(true_mask_binary.flatten())
            y_pred.extend(pred_mask_binary.flatten())
            
            accuracy_metric.update_state(true_mask_binary.flatten(), pred_mask_binary.flatten())
            precision_metric.update_state(true_mask_binary.flatten(), pred_mask_binary.flatten())
            recall_metric.update_state(true_mask_binary.flatten(), pred_mask_binary.flatten())
            iou_metric.update_state(true_mask_binary.flatten(), pred_mask_binary.flatten())
        
        # Calculate metrics
        accuracy = accuracy_metric.result().numpy()
        precision = precision_metric.result().numpy()
        recall = recall_metric.result().numpy()
        iou = iou_metric.result().numpy()
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Create a DataFrame to display the metrics in a table format
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy','Precision', 'Recall', 'F1-Score', 'IoU'],
            'Score': [accuracy, precision, recall, f1, iou]
        })
        
        print(metrics_df)
