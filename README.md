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
1. Split records into train/val/test sets
1. Convert raw data to tf records
    - `./src/convert_to_tfrecords.py -inputdir <input> -output <output>`
1. Run the training script
    - `./src/train_model.py -inputdir <input> -modelfile <output>`
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

# References
+ donkeycar **TODO**
+ tensorflow records **TODO**
