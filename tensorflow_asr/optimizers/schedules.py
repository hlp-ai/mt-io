import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, max_lr=None):
        super(TransformerSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # lr = (d_model^-0.5) * min(step^-0.5, step*(warm_up^-1.5))
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        if self.max_lr is not None:
            return tf.math.minimum(self.max_lr, lr)
        return lr

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
        }


class CyclicTransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """This callback implements a cyclical learning rate policy (CLR) to the square
    root decay generally used to train transformers.
    The method cycles the learning rate around the square root decay LR with an amplitude
    equal to the target LR with a given period.
    # Arguments
        d_model: The dimension of the transformer model.
        warmup_steps: Warm up steps where the LR increases linearly.
            Default to 4000 steps.
        max_lr: Maximum value of the learning rate reachable.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.

    It is inspired from the paper:
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(self, d_model, warmup_steps=4000, max_lr=None, step_size=None):
        """Applies triangular cyclic to the square root decay learning rate.
        Args:
        d_model: Model dimension
        warmup_steps: Warm up steps where the LR increases linearly.
        max_lr: The maximum LR.
        step_size: The size of the cyclic triangular half cycle.
        """
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.step_size = tf.cast(step_size, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = step * (self.warmup_steps ** -1.5)
        lr = 2 * tf.math.rsqrt(step)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(lr, warmup)
        lr = tf.math.minimum(self.max_lr, lr)
        cycle = tf.math.floor(1 + step / (2 * self.step_size))
        x = tf.math.abs(step / self.step_size - 2 * cycle + 1)
        lr = lr * (0.5 + tf.math.maximum(0.0, x))
        lr = tf.math.minimum(self.max_lr, tf.math.minimum(lr, warmup))
        return lr

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "step_size": self.step_size,
        }
