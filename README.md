# Tools to train neural networks for self-driving
This repository provides tools to train neural networks in AWS. It helps with converting the data from raw format to tf records, data augmentation, training and evaluation and keeping track of experiments.

## Getting started
### Installation
1. Get the codes
    - `git clone <this-repository`
1. Get into the codes
    - `cd this-repository`
1. Install dependencies

## Usage / pipeline / steps
1. Upload your data to S3 **OPTIONAL**
1. (Clean the data **TODO**)
1. Split records into train/val/test setss
    ```
    python3 src/data_split.py   -inputdir <str> 
                                -tub-dir-prefix <str>
                                -outputdir <str>
                                -train-split <float 0.0-1.0>
                                -val-split <float 0.0-1.0>
                                -test-split <float 0.0-1.0>
    ```
1. Convert raw data to tf records
    ```
    python3 src/convert_to_tfrecords.py -inputdir <str> -outputdir <str>
    ```
1. Run the training script
    ```
    # Example:
    python3 src/train_model.py -train-dir <str> -val-dir <str>

    usage: Train model [-h] [-train-dir TRAIN_DIR] [-val-dir VAL_DIR]
                   [-test-dir TEST_DIR] [-output-model-file OUTPUT_MODEL_FILE]
                   [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                   [--min-delta MIN_DELTA] [--patience PATIENCE]
                   [--input-image-width INPUT_IMAGE_WIDTH]
                   [--input-image-height INPUT_IMAGE_HEIGHT]
                   [--input-image-channels INPUT_IMAGE_CHANNELS]
                   [--input-image-vertical-crop-pixels INPUT_IMAGE_VERTICAL_CROP_PIXELS]
                   [--weight-angle-loss WEIGHT_ANGLE_LOSS]
                   [--weight-throttle-loss WEIGHT_THROTTLE_LOSS]
                   [--verbose VERBOSE]
    ```
1. Monitor
    - **TODO**
    - Tensorboard / Valohai / AWS Sagemaker
1. Evaluate
    - **TODO**
    - Mlflow / Valohai / AWS Sagemaker
1. Compare / benchmark models
    - **TODO**
    - Mlfow / Valohai
1. Deployment
    - **TODO**
    - `aws s3 cp <model> <car>` :)

# Sponsors
[spiceprogram.org](https://spiceprogram.fi) / [futurice.com](https://futurice.com)

# References
+ donkeycar **TODO**
+ tensorflow records **TODO**
+ that good tensorflow records tutorial **TODO**
