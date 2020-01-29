import os
import glob
import json
import shutil

import time

import tensorflow as tf

from utils import tf_tools

def donkey_car_load_records_from_dir(inputdir: str, tubdir: str) -> list:
    'Loads donkeycar type of json records from a directory'

    def _parse_record_index(filename: str) -> int:
        return int(filename.split('_')[1].split('.')[0])

    # Get a list of records
    filenames = glob.glob(os.path.join(os.path.join(inputdir, tubdir, 'record_*.json')))

    # Sort them based on indexes
    idx = [-1] * len(filenames)
    for i, filename in enumerate(filenames):
        idx[i] = _parse_record_index(os.path.basename(filename))
    idx.sort()

    # Load datas from json records
    records = [dict()] * len(filenames)
    # TODO: Speed up the loading. This loop takes agest (>90% of time)
    for i, ix in enumerate(idx):
        with open(os.path.join(inputdir, tubdir, 'record_%d.json' % ix), 'r') as f:
            records[i] = json.load(f)
            records[i].update({'img_path': os.path.join(tubdir, records[i]['cam/image_array'])})

    return records

def process_donkey_data_dir(inputdir: str, tub_dir_prefix: str = '') -> dict:
    
    session_dirs = glob.glob(os.path.join(inputdir, tub_dir_prefix + '*'))

    session_records = dict()
    for session_dir in session_dirs:
        # Check that the dir is actually a directory
        if os.path.isdir(session_dir):
            session_name = session_dir[len(inputdir):]
            session_records[session_name] = donkey_car_load_records_from_dir(inputdir=inputdir, tubdir=session_name)
            # If there is no records, lets remove the session
            if len(session_records[session_name]) == 0:
                _ = session_records.pop(session_name)
    
    return session_records

def move_train_split_files(inputdir: str, outputdir: str, train_records: list, val_records: list = None, test_records: list = None):

    def _move_records_data(inputdir: str, outputdir: str, records: list):
        if os.path.exists(outputdir) == False:
            os.makedirs(outputdir)

        with open(os.path.join(outputdir, 'records.csv'), 'w') as f_csv:
            f_csv.write("'cam/image_array','timestamp','user/throttle','user/angle','user/mode','img_path'\n")
            for record in records:

                img_file = record['img_path']
                if os.path.exists(os.path.dirname(os.path.join(outputdir, img_file))) == False:
                    os.makedirs(os.path.dirname(os.path.join(outputdir, img_file)))
                
                shutil.copyfile(os.path.join(inputdir, img_file), os.path.join(outputdir, img_file))
                f_csv.write('%s,%s,%f,%f,%s,%s\n' % (
                    record['cam/image_array'],
                    record['timestamp'], 
                    record['user/throttle'],
                    record['user/angle'],
                    record['user/mode'],
                    record['img_path']))

    _move_records_data(inputdir=inputdir, outputdir=os.path.join(outputdir, 'train'), records=train_records)
    _move_records_data(inputdir=inputdir, outputdir=os.path.join(outputdir, 'val'), records=val_records)
    _move_records_data(inputdir=inputdir, outputdir=os.path.join(outputdir, 'test'), records=test_records)


def load_tub_data_to_records(data_dir):
    'Old tub loading function. Works still and provides a baseline'
    # Get a list of directories starting with word tub
    tub_dirs = glob.glob(os.path.join(data_dir, 'tub*'))
    # Sort the directories
    tub_dirs.sort()
    tub_dirs = [tub_dir for tub_dir in tub_dirs]
    print(tub_dirs)
    # Go through the directories 
    records = []
    for tub_dir in tub_dirs:
        json_files = glob.glob(os.path.join(tub_dir, 'record_*.json'))
        if len(json_files) == 0:
            tub_dir = os.path.join(tub_dir, 'tub')
            json_files = glob.glob(os.path.join(tub_dir, 'record_*.json'))
        n = len(json_files)
        i = 0
        cnt = 0
        while cnt < n:
            json_file = os.path.join(tub_dir, 'record_%d.json' % i)
            try:
                data = json.load(open(json_file, 'r'))
                data['img_path'] = os.path.join(os.path.basename(tub_dir), data['cam/image_array'])
                records.append(data)
                cnt += 1
            except:
                pass
            i += 1

    return records

def load_csv_records(filename: str) -> list():
    'Loads a csv file and returns a list'

    records = []

    with open(filename, 'r') as f:
        # Read header
        columns = f.readline().replace("'",'').replace('\n', '').split(',')

        for line in f:
            values = line.replace('\n', '').split(',')
            #all_values.append(values)
            # 'cam/image_array','timestamp','user/throttle','user/angle','user/mode','img_path'
            record = dict()
            for i, column in enumerate(columns):
                if (column == 'user/throttle') or (column == 'user/angle'):
                    values[i] = float(values[i])
                record[column] = values[i]

            records.append(record)
    return records

def decode_img(img: bytes):

    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [120, 180])

def convert_data_to_tfrecords(inputdir: str, output: str):
    #records = load_tub_data_to_records(inputdir)
    records = load_csv_records(os.path.join(inputdir, 'records.csv'))

    # Make sure that dest dir exists
    if os.path.exists(os.path.dirname(output)) == False:
        os.makedirs(os.path.dirname(output))

    # Write the `tf.Example` observations to the file.
    with tf.io.TFRecordWriter(output) as writer:
        for i, record in enumerate(records):
            # parse fields
            image_string = open(os.path.join(inputdir, record['img_path']), 'rb').read()
            angle = record['user/angle']
            throttle = record['user/throttle']
            example = tf_tools.serialize_example(image_string, angle, throttle)
            writer.write(example)
            
            if i % 1000 == 0:
                print(i, len(records), 100*i/len(records))