import os
import sys
import argparse
import glob
import logging
import json
import datetime

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from utils import fileio
from utils import model_selection
from utils import tf_tools
from models import basic_linear_model, mobilenet_v2
import defaults as defaults


def parse_arguments(argv):

    parser = argparse.ArgumentParser(prog="Train model")
    parser.add_argument("-train-dir", type=str)
    parser.add_argument("-val-dir", type=str, default="")
    parser.add_argument("-test-dir", type=str, default="")
    parser.add_argument("-output-model-file", type=str, default="data/models/model.h5")
    parser.add_argument("--model-arch-name", type=str, default="basiclinear")
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


def main(argv):

    args = parse_arguments(argv=argv)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Load a bunch of training records
    logger.info("Loading training samples tfrecords")
    parsed_trainset = fileio.read_tfrecords_dir(
        dirname=args.train_dir,
        image_width=args.input_image_width,
        image_height=args.input_image_height,
        image_channels=args.input_image_channels,
    )

    # Read validation dataset
    logger.info("Loading validation samples tfrecords")
    parsed_validationset = fileio.read_tfrecords_dir(
        dirname=args.val_dir,
        image_width=args.input_image_width,
        image_height=args.input_image_height,
        image_channels=args.input_image_channels,
    )

    num_train_samples = sum(1 for record in parsed_trainset)
    num_val_samples = sum(1 for record in parsed_validationset)

    print(
        "Number of training / validation samples %d/%d"
        % (num_train_samples, num_val_samples)
    )

    # Initialise model with given architecture
    if args.model_arch_name == "basiclinear":
        model = basic_linear_model.create_model(
            image_width=args.input_image_width,
            image_height=args.input_image_height,
            image_channels=args.input_image_channels,
            crop_margin_from_top=args.input_image_vertical_crop_pixels,
        )
    elif args.model_arch_name == "mobilenetv2":
        model = mobilenet_v2.create_model(
            image_width=args.input_image_width,
            image_height=args.input_image_height,
            image_channels=args.input_image_channels,
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

    log_dir = "logs/fit/" + args.model_arch_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir, histogram_freq=1)

    # Train the car
    model.fit(
        parsed_trainset,
        validation_data=parsed_validationset,
        steps_per_epoch=num_train_samples // args.batch_size,
        validation_steps=num_val_samples // args.batch_size,
        epochs=args.epochs,
        callbacks=[tf_tools.JsonLogger(), save_best, early_stop, tensorboard],
        verbose=args.verbose,
    )


if __name__ == "__main__":

    main(sys.argv)
