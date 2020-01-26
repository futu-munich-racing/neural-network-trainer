import sys
import argparse

from utils import fileio

def parse_input_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputdir', type=str)
    parser.add_argument('-outputdir', type=str)
    parser.add_argument('-tub-dir-prefix', type=str, default='')
    parser.add_argument('-train-split', type=float, default=0.7)
    parser.add_argument('-val-split', type=float, default=0.15)
    parser.add_argument('-test-split', type=float, default=0.15)
    
    args = parser.parse_args()

    # Normalize train/val/test split in case they don't sum up to one
    # TODO: We could have some more intelligent way to make sure sum does not over 1.
    #   - Fix the given values. Use as much is left from the rest.
    #   - If all value are give e.g. 70, 15 15 then just perform the normalisation like now
    split_sum = args.train_split + args.val_split + args.test_split
    args.train_split /= split_sum
    args.val_split /= split_sum
    args.test_split /= split_sum

    return args

def main(argv):

    # Read input arguments
    args = parse_input_arguments(argv)

    print(args)

    # Read recorded data
    records = fileio.process_donkey_data_dir(inputdir=args.inputdir, tub_dir_prefix=args.tub_dir_prefix)

    # Split the data
    train_records, val_records, test_records = model_selection.train_val_test_split_records(records,
                                                                                            args.train_split,
                                                                                            args.val_split,
                                                                                            args.test_split)
    # Generate split datasets
    move_train_split_files()

if __name__ == "__main__":
    main(sys.argv)