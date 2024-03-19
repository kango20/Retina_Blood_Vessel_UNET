# import libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array

class Viz:
    def __init__(self, RetinaDatasetPreprocessor, Model):
        self.train_images = RetinaDatasetPreprocessor.train
        self.train_masks = RetinaDatasetPreprocessor.train_mask
        self.test_images = RetinaDatasetPreprocessor.test
        self.test_masks = RetinaDatasetPreprocessor.test_mask
        self.model = Model.model
        self.history = Model.history 
        
    # Display a set of random images along with their corresponding ground truth masks.
    def display_images_with_masks(self, num_images=10):
        indices = np.random.choice(np.arange(len(self.train_images)), size=num_images, replace=False)

        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))
        for i, idx in enumerate(indices):
            image = self.train_images[idx]
            mask = self.train_masks[idx].squeeze() 

            if num_images == 1:
                axes = np.array([[axes]])

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {idx}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'Mask {idx}')
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig('/app/rundir/Retina_Blood_Vessel_GroundTruthMasks.png')


    # Display a set of random images along with the model's predicted masks.
    def display_images_with_predictions(self, num_images=10):
        indices = np.random.choice(np.arange(len(self.train_images)), size=num_images, replace=False)
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))
        for i, idx in enumerate(indices):
            image = self.train_images[idx]
            mask = self.train_masks[idx].squeeze()  

            # Predict and process the mask
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))[0]
            pred_mask = pred_mask.squeeze()  
            # Remove batch dimension and channel dimension if 1
            if num_images == 1:
                axes = np.array([[axes]])
            
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {idx}')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(pred_mask > 0.5, cmap='gray')  
            axes[i, 1].set_title(f'Predicted Mask {idx}')
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig('/app/rundir/Retina_Blood_Vessel_PredictedMasks.png')

    # Displays the precision recall curve
    def plot_precision_recall_curve(self):
        y_true = []
        y_scores = []

        for i in range(len(self.test_images)):
            image = self.test_images[i]
            # Binary classification
            true_mask = self.test_masks[i].squeeze()

            # Predicts the mask
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))
            pred_mask = pred_mask.squeeze()

            # Flatten
            y_true.extend(true_mask.flatten())
            y_scores.extend(pred_mask.flatten())

        # Makes sure y_true is binary
        y_true = np.array(y_true).astype(np.int32)
        y_scores = np.array(y_scores)

        # Calculates precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        plt.figure()
        plt.step(recall, precision, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
        plt.savefig('/app/rundir/Precision_Recall_Curve.png')

    # Plotting the IoU scores in histograms
    def plot_iou_scores(self):
        iou_scores = []

        for image, mask in zip(self.test_images, self.test_masks):
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))[0].squeeze()

            # Initialize a new IoU metric object for each image and mask pair
            iou_metric = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
            iou_metric.update_state(np.round(mask).astype(int), np.round(pred_mask).astype(int))

            # Append the IoU score for the current image and mask pair to the list
            iou_scores.append(iou_metric.result().numpy())

        plt.figure(figsize=(10, 6))
        plt.hist(iou_scores, bins=20, color='skyblue', edgecolor='black')
        plt.title('Histogram of IoU Scores')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('/app/rundir/IoU_Histogram.png')

    # Display error maps for a number of random images 
    def display_error_maps(self, num_images = 10):    
        indices = np.random.choice(np.arange(len(self.test_images)), size=num_images, replace=False)
        # increasing the figure size for better visualizations
        fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5)) 
        for i, idx in enumerate(indices):
            image = self.test_images[idx]
            true_mask = self.test_masks[idx].squeeze()
            pred_mask = self.model.predict(np.expand_dims(image, axis = 0))[0].squeeze()
            error_map = np.abs(true_mask - pred_mask)
            
            # makes axis 2D
            if num_images == 1:
                axes = axes.reshape(-1,3)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {idx}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask, cmap="gray")
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(error_map, cmap = 'hot', interpolation = 'nearest')
            axes[i, 2].set_title("Error Map")
            axes[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig('/app/rundir/ErrorMaps.png')

    # Display feature maps for a random image from the test set and a random layer.
    def display_feature_maps(self, num_columns=6):
        # Randomly select an image from the test set
        idx = np.random.choice(len(self.test_images))
        image_to_visualize = self.test_images[idx]

        # Choose a random convolutional layer from the model
        conv_layers = [layer for layer in self.model.layers if 'conv' in layer.name]
        if not conv_layers:
            print("No convolutional layers found in the model.")
            return
        layer_to_visualize = np.random.choice(conv_layers)

        # Define a Model that will return these outputs, given the model input
        feature_map_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=layer_to_visualize.output)

        # Get the feature maps
        feature_maps = feature_map_model.predict(np.expand_dims(image_to_visualize, axis=0))

        num_features = feature_maps.shape[-1] 
        num_rows = np.ceil(num_features / num_columns).astype(int)

        plt.figure(figsize=(num_columns * 2, num_rows * 2))
        for i in range(num_features):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f'Feature maps from layer: {layer_to_visualize.name}')
        plt.savefig('/app/rundir/FeatureMaps.png')

    # Plot a confusion matrix for binary segmentation tasks.
    def plot_confusion_matrix(self):
        y_true = []
        y_pred = []

        for image, mask in zip(self.test_images, self.test_masks):
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))[0].squeeze()
            binary_pred_mask = np.round(pred_mask).astype(int)

            y_true.extend(mask.flatten())
            y_pred.extend(binary_pred_mask.flatten())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_mtx, annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.savefig('/app/rundir/ConfusionMatrix.png')

    # Plot the training and validation loss curves.
    def plot_loss_curve(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('/app/rundir/LossCurve.png')

    # Display the 3 best and 3 worst segmentation results based on IoU scores
    def display_best_worst(self):
        iou_scores = []
        images = []
        pred_masks = []
        true_masks = []

        # Calculate IoU for each image and mask pair
        for image, mask in zip(self.test_images, self.test_masks):
            pred_mask = self.model.predict(np.expand_dims(image, axis=0))[0].squeeze()
            iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
            iou_metric.update_state(np.round(mask).astype(int), np.round(pred_mask).astype(int))
            iou_score = iou_metric.result().numpy()

            iou_scores.append(iou_score)
            images.append(image)
            pred_masks.append(pred_mask)
            true_masks.append(mask.squeeze())

        # Rank the results by IoU score
        sorted_indices = np.argsort(iou_scores)
        # 3 best results
        top_indices = sorted_indices[-3:]
        # 3 worst results  
        bottom_indices = sorted_indices[:3]  
        combined_indices = np.concatenate([top_indices, bottom_indices])

        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(combined_indices):
            plt.subplot(6, 3, 3*i+1)
            plt.imshow(images[idx])
            plt.title(f"Image {idx}")
            plt.axis('off')

            plt.subplot(6, 3, 3*i+2)
            plt.imshow(true_masks[idx], cmap='gray')
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(6, 3, 3*i+3)
            plt.imshow(pred_masks[idx] > 0.5, cmap='gray')  # Apply threshold
            plt.title("Predicted Mask")
            plt.axis('off')

        plt.tight_layout()
        plt.suptitle("3 Best (Top Rows) and 3 Worst (Bottom Rows) Results based on IoU")
        plt.savefig('/app/rundir/3BestWorst.png')
