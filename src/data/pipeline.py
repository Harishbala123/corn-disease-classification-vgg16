"""tf.data input pipeline with augmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def make_dataset(paths: list[str], labels: np.ndarray, n_classes: int, batch: int, crop: int, training: bool,
                 seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), labels.astype(np.int32)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, y):
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.cast(img, tf.float32)
        if training:
            img = tf.image.random_crop(img, [crop, crop, 3])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
            img = tf.image.random_brightness(img, 25.0)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 255.0)
        else:
            off = (tf.shape(img)[0] - crop) // 2
            img = img[off:off + crop, off:off + crop, :]
        return img, {"probs": tf.one_hot(y, n_classes), "embedding": y}

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def full_paths(processed_dir: Path, rel: list[str]) -> list[str]:
    return [str(Path(processed_dir) / r) for r in rel]
