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
