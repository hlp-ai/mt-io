import tensorflow as tf


# Create a dictionary describing the features.
_image_feature_description = {
    'img': tf.io.FixedLenFeature([], tf.string),
    'rpn_class': tf.io.FixedLenFeature([], tf.string),
    'rpn_regress': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, _image_feature_description)


def _parse_tensor(e):
    return tf.io.parse_tensor(e["img"], out_type=tf.float32), {"rpn_class": tf.io.parse_tensor(e["rpn_class"], out_type=tf.float32),
            "rpn_regress": tf.io.parse_tensor(e["rpn_regress"], out_type=tf.float32)}


def get_ctpn_ds(tfds_file):
    raw_image_dataset = tf.data.TFRecordDataset(tfds_file)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    # for e in parsed_image_dataset.take(1):
    #     print(e.keys())

    ds = parsed_image_dataset.map(_parse_tensor).prefetch(64)
    print(ds.element_spec)
    # for e in ds.take(1):
    #     print(e["img"])

    return ds


if __name__ == "__main__":
    ds = get_ctpn_ds("./ctpn.tfrecords")
    # for m, d in ds:
    #     print(m.shape, d["rpn_class"].shape, d["rpn_regress"].shape)
