import tensorflow as tf


class AugmentationMethod:
    @tf.function
    def augment(self, *args, **kwargs):
        raise NotImplementedError()
