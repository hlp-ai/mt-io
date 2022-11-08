import tensorflow as tf


class ErrorRate(tf.keras.metrics.Metric):
    """Metric for WER or CER"""

    def __init__(
        self,
        func,
        name="error_rate",
        **kwargs,
    ):
        super(ErrorRate, self).__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name=f"{name}_numerator", initializer="zeros")
        self.denominator = self.add_weight(name=f"{name}_denominator", initializer="zeros")
        self.func = func

    def update_state(
        self,
        decode: tf.Tensor,
        target: tf.Tensor,
    ):
        n, d = self.func(decode, target)
        self.numerator.assign_add(n)
        self.denominator.assign_add(d)

    def result(self):
        return tf.math.divide_no_nan(self.numerator, self.denominator)
