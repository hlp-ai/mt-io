import abc
import tensorflow as tf

from tensorflow_asr.augmentations.augmentation import Augmentation

BUFFER_SIZE = 100
TFRECORD_SHARDS = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE


class BaseDataset(metaclass=abc.ABCMeta):
    """Based dataset for all models"""

    def __init__(
        self,
        data_paths: list,
        augmentations: Augmentation = Augmentation(None),
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        use_tf: bool = False,
        stage: str = "train",
        **kwargs
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.augmentations = augmentations  # apply augmentation
        self.cache = cache  # whether to cache transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size  # shuffle buffer size
        self.stage = stage  # for defining tfrecords files
        self.use_tf = use_tf
        self.drop_remainder = drop_remainder  # whether to drop remainder for multi gpu training
        self.indefinite = indefinite  # Whether to make dataset repeat indefinitely -> avoid the potential last partial batch
        self.total_steps = None  # for better training visualization

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, batch_size):
        raise NotImplementedError()
