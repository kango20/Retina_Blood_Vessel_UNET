# import libraries
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models
import keras


# Set seeds to get standard results
keras.utils.set_random_seed(44)
tf.config.experimental.enable_op_determinism()

class Model:
    def __init__(self, train, train_mask, test, test_mask) -> None:
        self.train = train
        self.train_mask = train_mask
        self.test = test
        self.test_mask = test_mask
        self.model = self.create_model(input_shape=(512, 512, 3))  # Specify the input shape explicitly if needed
        self.history = None  

    # Creates the convolution blocks
    def conv_block(self, inputs, num_filters):
        x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    # Creates the encoder blocks using the convolution blocks
    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters)
        p = layers.MaxPooling2D((2, 2))(x)
        return x, p

    # Creates decoder blocks with transpose convolutions
    def decoder_block(self, inputs, skip_features, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    # Assembles the U-Net architecture
    def build_unet(self, input_shape):
        inputs = layers.Input(shape=input_shape)

        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        b = self.conv_block(p4, 1024)  # Bottleneck to connect encoder and decoder pathways

        d1 = self.decoder_block(b, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

        model = models.Model(inputs, outputs, name="U-Net")
        return model

    # Creates the model
    def create_model(self, input_shape=(512, 512, 3)):
        model = self.build_unet(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    # Training and fitting the model
    def train_model(self):
        self.history = self.model.fit(
            x=self.train,
            y=self.train_mask,
            batch_size=2,  
            epochs=50, 
            validation_split=0.2,
            shuffle=True
        )
        return self.history
