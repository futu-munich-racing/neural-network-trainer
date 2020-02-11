import os
import sys
import argparse
import glob
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
    parser.add_argument(
        "--input-image-width", type=int, default=defaults.INPUT_IMAGE_WIDTH
    )
    parser.add_argument(
        "--input-image-height", type=int, default=defaults.INPUT_IMAGE_HEIGHT
    )
    parser.add_argument(
        "--input-image-channels", type=int, default=defaults.INPUT_IMAGE_CHANNELS
    )
    parser.add_argument(
        "--input-image-vertical-crop-pixels",
        default=defaults.INPUT_IMAGE_CROP_VERTICAL_PIXELS,
    )
    parser.add_argument(
        "--weight-angle-loss", type=float, default=defaults.WEIGHT_ANGLE_LOSS
    )
    parser.add_argument(
        "--weight-throttle-loss", type=float, default=defaults.WEIGHT_THROTTLE_LOSS
    )
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    return args


def _parse_fn(
    example_serialized, is_training=False, img_width=256, img_height=256, img_channels=3
):
    """ Parse tensorflow records and return X, y, 
        where X is image and y is (angle and throttle)
    """
    # TODO: it would be cool, if this could come from a file e.g. meta.json (donkeycar)
    feature_map = {
        "image": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image_rows": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "image_cols": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "angle": tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.0),
        "throttle": tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.0),
    }

    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = tf.io.decode_jpeg(parsed["image"])
    print(image.shape)
    print(parsed)
    img_height, img_width, img_channels = (parsed["image_rows"], parsed["image_cols"], 3) #tf.shape(image).numpy()
    print(f'image shape: {img_height}x{img_width}x{img_channels}')
    image = tf.reshape(image, (1, img_height, img_width, img_channels))
    return (image, (parsed["angle"], parsed["throttle"]))


class JsonLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(json.dumps(dict({"loss": logs["loss"], "val_loss": logs["val_loss"]})))

def read_tfrecords_dir(dirname: str, image_width: int=256, image_height: int=256, image_channels: int=3):
    filenames = glob.glob(os.path.join(dirname, "*.tfrecord"))
    
    print(f'tfrecords: {filenames}')

    raw_dataset = tf.data.TFRecordDataset(
        filenames=filenames
    )

    dataset = raw_dataset.map(
        lambda d: _parse_fn(
            d,
            image_width,
            image_height,
            image_channels,
        )
    )

    return dataset

def main(argv):

    args = parse_arguments(argv=argv)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Load a bunch of training records
    logger.info('Loading training tf-records')
    parsed_trainset = read_tfrecords_dir(
        args.train_dir,
        args.input_image_width,
        args.input_image_height,
        args.input_image_channels
        )

    # Read validation dataset
    parsed_validationset = read_tfrecords_dir(
        args.val_dir,
        args.input_image_width,
        args.input_image_height,
        args.input_image_channels
        )

    num_train_samples = sum(1 for record in parsed_trainset)
    num_val_samples = sum(1 for record in parsed_validationset)

    print(
        "Number of training / validation samples %d/%d"
        % (num_train_samples, num_val_samples)
    )

    model = basic_linear_model.create_model(
        img_dims=[
            256, #args.input_image_height,
            341, #args.input_image_width,
            args.input_image_channels,
        ],
        crop_margin_from_top=args.input_image_vertical_crop_pixels,
    )

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
        verbose=args.verbose,
    )


if __name__ == "__main__":

    main(sys.argv)
