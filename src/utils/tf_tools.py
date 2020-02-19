import json
import tensorflow as tf

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, angle, throttle):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        "image": _bytes_feature(image),
        "angle": _float_feature(angle),
        "throttle": _float_feature(throttle),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
  Args:
  image_buffer: scalar string Tensor.
  scope: Optional scope for name_scope.
  Returns:
  3-D float Tensor with values ranging from [0, 1).
  """
    # with tf.name_scope(values=[image_buffer], name=scope,
    #                default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height
    # and width that is set dynamically by decode_jpeg. In other
    # words, the height and width of image is unknown at compile-i
    # time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).
    # The various adjust_* ops all require this range for dtype
    # float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def _parse_fn(
    example_serialized, is_training=False, img_width=256, img_height=256, img_channels=3
):
    """ Parse tensorflow records and return X, y, 
        where X is image and y is (angle and throttle)
    """
    # TODO: it would be cool, if this could come from a file e.g. meta.json (donkeycar)
    feature_map = {
        "image": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "angle": tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.0),
        "throttle": tf.io.FixedLenFeature([], dtype=tf.float32, default_value=0.0),
    }

    # Parse Example / sample in tfrecord
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    # Decode JPEG compressed image
    image = tf.io.decode_jpeg(parsed["image"])
    # Resize image to given size
    image = tf.image.resize(image, (img_height, img_width))
    # Reshape image from 3D to 4D
    image = tf.reshape(image, (1, img_height, img_width, img_channels))
    return (image, (parsed["angle"], parsed["throttle"]))


class JsonLogger(tf.keras.callbacks.Callback):
    "Simple JSON Logger: Prints metrics as JSON so that it can be monitored while training."

    def on_epoch_end(self, epoch, logs: dict = None):
        def _convert_values_to_floats(logs: dict):
            "Convert dictionary values to floats fron numpy float32"
            for item in logs.items():
                logs[item[0]] = item[1].astype("float")
            return logs

        print(json.dumps(_convert_values_to_floats(logs)))
