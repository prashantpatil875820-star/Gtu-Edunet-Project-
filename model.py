"""
CNN architecture for forest-fire risk classification from satellite imagery.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

import config


def build_cnn(input_shape=None, num_classes=None):
    """
    A compact but regularized CNN suitable for satellite image patches.
    Architecture: 4 conv blocks (Conv->BN->ReLU->Pool) + GAP + dense head.
    Using GlobalAveragePooling instead of Flatten keeps the parameter count
    low and reduces overfitting on modest-sized satellite datasets.
    """
    if input_shape is None:
        input_shape = (*config.IMAGE_SIZE, config.CHANNELS)
    if num_classes is None:
        num_classes = (
            len(config.CLASS_NAMES_MULTI)
            if config.USE_MULTICLASS
            else 1  # binary -> single sigmoid output
        )

    inputs = layers.Input(shape=input_shape)

    # Built-in augmentation layers (active only during training)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # Rescale [0,255] -> [0,1]
    x = layers.Rescaling(1.0 / 255)(x)

    def conv_block(x, filters, l2=1e-4):
        x = layers.Conv2D(filters, 3, padding="same",
                           kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same",
                           kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        return x

    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="fire_risk")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"),
                   tf.keras.metrics.Precision(name="precision"),
                   tf.keras.metrics.Recall(name="recall")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="fire_risk")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(inputs, outputs, name="ForestFireRiskCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=metrics,
    )
    return model


def build_transfer_model(num_classes=None, backbone="MobileNetV2", fine_tune_at=100):
    """
    Optional stronger alternative: transfer learning on a pretrained backbone.
    Recommended once you have >2-3k labeled satellite patches, since it
    converges faster and generalizes better than the scratch CNN above.
    """
    if num_classes is None:
        num_classes = (
            len(config.CLASS_NAMES_MULTI)
            if config.USE_MULTICLASS
            else 1
        )
    input_shape = (*config.IMAGE_SIZE, config.CHANNELS)

    if backbone == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet")
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet")
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="fire_risk")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="fire_risk")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(inputs, outputs, name=f"ForestFireRisk_{backbone}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss=loss,
        metrics=metrics,
    )
    return model


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
