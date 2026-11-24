"""VGG16 transfer-learning model with a classification head and an embedding head."""
from __future__ import annotations

import keras
from keras import layers, ops

VGG_MEAN_BGR = [103.939, 116.779, 123.68]


@keras.saving.register_keras_serializable(package="corn")
class VGGPreprocess(layers.Layer):
    """Same as keras.applications.vgg16.preprocess_input (RGB 0-255 -> BGR, ImageNet mean subtracted),
    as a serialisable layer so the saved model accepts plain RGB images."""

    def call(self, x):
        x = ops.flip(x, axis=-1)
        return x - ops.convert_to_tensor(VGG_MEAN_BGR, dtype=x.dtype)


def build_model(n_classes: int, crop: int = 224, embedding_dim: int = 128, dense_units: int = 256,
                dropout: float = 0.5, weights: str | None = "imagenet") -> keras.Model:
    inp = keras.Input((crop, crop, 3), name="image")
    x = VGGPreprocess(name="vgg_preprocess")(inp)
    base = keras.applications.VGG16(include_top=False, weights=weights, input_shape=(crop, crop, 3))
    base._name = "vgg16"
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    emb = layers.Dense(embedding_dim, name="embedding_dense")(x)
    emb_norm = layers.UnitNormalization(name="embedding")(emb)
    probs = layers.Dense(n_classes, activation="softmax", name="probs")(layers.Activation("relu")(emb))
    return keras.Model(inp, {"probs": probs, "embedding": emb_norm}, name="vgg16_corn")


def set_trainable(model: keras.Model, stage: str) -> None:
    """stage 'head': VGG16 frozen; stage 'finetune': block5 trainable."""
    base = model.get_layer("vgg16")
    base.trainable = stage == "finetune"
    if stage == "finetune":
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")
