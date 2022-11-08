import tensorflow as tf


# TODO: clean mixed precision API when TensorFlow requirement is updated to >=2.4.
_set_global_policy = tf.keras.mixed_precision.experimental.set_policy

_get_global_policy = tf.keras.mixed_precision.experimental.global_policy


def enable_mixed_precision(force=False):
    """Globally enables mixed precision if the detected hardware supports it.

    Args:
      force: Set ``True`` to force mixed precision mode even if the hardware
        does not support it.

    Returns:
      A boolean to indicate whether mixed precision was enabled or not.
    """
    if not force:
        gpu_devices = tf.config.get_visible_devices("GPU")
        if not gpu_devices:
            tf.get_logger().warning("Mixed precision not enabled: no GPU is detected")
            return False

        gpu_details = tf.config.experimental.get_device_details(gpu_devices[0])
        compute_capability = gpu_details.get("compute_capability")
        if compute_capability is None:
            tf.get_logger().warning(
                "Mixed precision not enabled: a NVIDIA GPU is required"
            )
            return False
        if compute_capability < (7, 0):
            tf.get_logger().warning(
                "Mixed precision not enabled: a NVIDIA GPU with compute "
                "capability 7.0 or above is required, but the detected GPU "
                "has compute capability %d.%d" % compute_capability
            )
            return False

    _set_global_policy("mixed_float16")
    return True


def disable_mixed_precision():
    """Globally disables mixed precision."""
    _set_global_policy("float32")


def mixed_precision_enabled():
    """Returns ``True`` if mixed precision is enabled."""
    policy = _get_global_policy()
    return "float16" in policy.name


def mixed_precision_wrapper(optimizer):
    # TODO: clean mixed precision API when TensorFlow requirement is updated to >=2.4.
    wrapper_class = tf.keras.mixed_precision.experimental.LossScaleOptimizer
    wrapper_kwargs = dict(loss_scale="dynamic")
    if not isinstance(optimizer, wrapper_class):
        optimizer = wrapper_class(optimizer, **wrapper_kwargs)

    return optimizer
