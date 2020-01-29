import sys
import argparse

IMG_DEFAULT_WIDTH = 256
IMG_DEFAULT_HEIGTH = 256

from utils import fileio

def parse_input_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputdir', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-imgwidth', type=int, default=IMG_DEFAULT_WIDTH)
    parser.add_argument('-imgheight', type=int, default=IMG_DEFAULT_HEIGTH)
    
    return parser.parse_args()

def convert_rawdat_to_tfrecords(inputdir: str, output: str, img_width: int=256, img_height: int=256):
    fileio.convert_data_to_tfrecords(inputdir, output)

def main(argv):
    args = parse_input_arguments(argv)
    print(args)

    convert_rawdat_to_tfrecords(inputdir=args.inputdir,
                                output=args.output)

if __name__ == "__main__":
    main(sys.argv)