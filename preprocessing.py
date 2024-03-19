import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class RetinaDatasetPreprocessor:
    def __init__(self, base_dir, augment_train=False):
        self.base_dir = base_dir
        self.augment_train = augment_train
        # Load datasets upon initialization with respective augmentations if necessary
        self.train, self.train_mask = self.load_dataset('train', self.augment_train)
        self.test, self.test_mask = self.load_dataset('test', False)  # No augmentation for test data

    # reads the image 
    def read_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # reads the images with the masks in greyscale
    def read_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask / 255.0
        return mask

    # loading the dataset into train or test
    def load_dataset(self, dataset_type='train', augment=False):
        images_dir = os.path.join(self.base_dir, dataset_type, 'image')
        masks_dir = os.path.join(self.base_dir, dataset_type, 'mask')

        images = []
        masks = []

        for img_file in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace('image', 'mask'))

            img = self.read_image(img_path)
            mask = self.read_mask(mask_path)

            images.append(img)
            masks.append(mask)

        if augment:
            images, masks = self.apply_augmentation(np.array(images), np.array(masks))

        return np.array(images), np.array(masks)

    def apply_augmentation(self, images, masks):
        # Define the same augmentation parameters for both train images and train masks
        data_gen_args = dict(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Creates an ImageDataGenerator for the images
        image_datagen = ImageDataGenerator(**data_gen_args)
        # Creates an ImageDataGenerator for the masks, with the same parameters
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Ensures same transformations 
        seed = 44

        # Create generators that will augment the images and masks
        image_generator = image_datagen.flow(images, batch_size=images.shape[0], seed=seed, shuffle=False)
        mask_generator = mask_datagen.flow(masks, batch_size=masks.shape[0], seed=seed, shuffle=False)

        # Generate augmented images and masks
        augmented_images = next(image_generator)
        augmented_masks = next(mask_generator)

        return augmented_images, augmented_masks

