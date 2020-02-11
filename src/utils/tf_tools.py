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
    image_rows, image_cols, image_channels = image.shape
    #image = tf.image.convert_image_dtype(image, tf.float32).numpy().tobytes()
    image = tf.image.encode_jpeg(image) #.numpy().tobytes() #.numpy()
    #image = image.numpy().tobytes()
    feature = {
        "image": _bytes_feature(image),
        "image_rows": _int64_feature(image_rows),
        "image_cols": _int64_feature(image_cols),
        "image_channels": _int64_feature(image_channels),
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
