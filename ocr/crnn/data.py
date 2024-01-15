import numpy as np
import tensorflow as tf
from keras import layers


# Desired image dimensions
img_width = 280
img_height = 32


class DataLoader:

    def __init__(self, meta_fn):
        self.images = []
        self.labels = []
        with open(meta_fn, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                img, label = line.split("\t")
                self.images.append(img)
                self.labels.append(label)

        characters = set(char for label in self.labels for char in label)
        self.characters = sorted(list(characters))

        print("Number of images found: ", len(self.images))
        print("Number of labels found: ", len(self.labels))
        print("Number of unique characters: ", len(self.characters))
        print("Characters present: ", self.characters)

        # Maximum length of any captcha in the dataset
        self.max_length = max([len(label) for label in self.labels])

        # Mapping characters to integers
        self.char_to_num = layers.StringLookup(vocabulary=list(self.characters), mask_token=None)

        # Mapping integers back to original characters
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def load_data(self, batch_size=64, train_size=0.95):
        # Splitting data into training and validation sets
        x_train, x_valid, y_train, y_valid = split_data(np.array(self.images), np.array(self.labels), train_size=train_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        train_dataset = (
            train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return train_dataset, validation_dataset

    def encode_single_sample(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_jpeg(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}


def split_data(images, labels, train_size=0.95, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


if __name__ == "__main__":
    data_loader = DataLoader(r"../../deva.txt")
    train_ds, valid_ds = data_loader.load_data()

    for e in train_ds.take(1):
        print(e)

    for e in valid_ds.take(1):
        print(e)
