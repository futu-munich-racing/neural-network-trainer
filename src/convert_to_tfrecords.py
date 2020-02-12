import sys
import argparse

from utils import fileio

import defaults


def parse_input_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputdir", type=str)
    parser.add_argument("-o", "--outputdir", type=str)
    parser.add_argument(
        "-n-images-per-file", type=int, default=defaults.N_IMAGES_PER_TFRECORD
    )

    return parser.parse_args()


def convert_rawdat_to_tfrecords(inputdir: str, output: str):
    fileio.convert_data_to_tfrecords(inputdir, output)


def main(argv):
    args = parse_input_arguments(argv)

    fileio.convert_data_to_tfrecords(
        inputdir=args.inputdir,
        outputdir=args.outputdir,
        n_images_per_file=args.n_images_per_file,
    )


if __name__ == "__main__":
    main(sys.argv)
