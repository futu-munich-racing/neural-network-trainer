import os
import sys
import argparse
import logging
import json

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import fileio
from utils import model_selection
from utils import tf_tools
from models import basic_linear_model
import defaults as defaults


def parse_arguments(argv):

    parser = argparse.ArgumentParser(prog="Train model")
    parser.add_argument("-train-dir", type=str)
    parser.add_argument("-val-dir", type=str, default="")
    parser.add_argument("-test-dir", type=str, default="")
    parser.add_argument("-output-model-file", type=str, default="data/models/model.h5")
    parser.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=defaults.EPOCHS)
    parser.add_argument("--min-delta", type=float, default=defaults.MIN_DELTA)
    parser.add_argument("--patience", type=int, default=defaults.PATIENCE)
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    return args


def _parse_fn(example_serialized, is_training=False):
    """ Parse tensorflow records and return X, y, 
        where X is image and y is (angle and throttle)
    """
    # TODO: it would be cool, if this could come from a file e.g. meta.json (donkeycar)
    feature_map = {
        "image": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "angle": tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.0),
        "throttle": tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.0),
    }

    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = tf.io.decode_jpeg(parsed["image"])
    # TODO: Image dims should also come from a config / input
    image = tf.reshape(image, (1, 240, 360, 3))
    return (image, (parsed["angle"], parsed["throttle"]))


class JsonLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(json.dumps(dict({"loss": logs["loss"], "val_loss": logs["val_loss"]})))


def main(argv):

    args = parse_arguments(argv=argv)

    raw_trainset = tf.data.TFRecordDataset(
        os.path.join(args.train_dir, "train.tfrecord")
    )
    parsed_trainset = raw_trainset.map(_parse_fn)

    # Read validation dataset
    raw_validationset = tf.data.TFRecordDataset(
        os.path.join(args.val_dir, "val.tfrecord")
    )
    parsed_validationset = raw_trainset.map(_parse_fn)

    num_train_samples = sum(1 for record in parsed_trainset)
    num_val_samples = sum(1 for record in parsed_validationset)

    print(
        "Number of training / validation samples %d/%d"
        % (num_train_samples, num_val_samples)
    )

    weight_loss_angle = 0.9
    weight_loss_throttle = 0.1

    model = basic_linear_model.create_model(img_dims=[240, 360, 3])

    if os.path.exists(os.path.dirname(args.output_model_file)) == False:
        os.makedirs(os.path.dirname(args.output_model_file))

    # checkpoint to save model after each epoch
    save_best = ModelCheckpoint(
        args.output_model_file,
        monitor="val_loss",
        verbose=args.verbose,
        save_best_only=True,
        mode="min",
    )

    # stop training if the validation error stops improving.
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=args.min_delta,
        patience=args.patience,
        verbose=args.verbose,
        mode="auto",
    )

    # Train the car
    model.fit(
        parsed_trainset,
        validation_data=parsed_validationset,
        steps_per_epoch=num_train_samples // args.batch_size,
        validation_steps=num_val_samples // args.batch_size,
        epochs=args.epochs,
        callbacks=[JsonLogger(), save_best, early_stop],
        verbose=0,
    )


if __name__ == "__main__":

    main(sys.argv)
