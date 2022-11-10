import fire
import tensorflow as tf

from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio


def main(
    filename: str,
    tflite: str = None,
    blank: int = 0,
    num_rnns: int = 1,
    nstates: int = 2,
    statesize: int = 320,
):
    tflitemodel = tf.lite.Interpreter(model_path=tflite)

    signal = read_raw_audio(filename)

    input_details = tflitemodel.get_input_details()
    output_details = tflitemodel.get_output_details()
    tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
    tflitemodel.allocate_tensors()
    tflitemodel.set_tensor(input_details[0]["index"], signal)
    tflitemodel.set_tensor(input_details[1]["index"], tf.constant(blank, dtype=tf.int32))
    tflitemodel.set_tensor(input_details[2]["index"], tf.zeros([num_rnns, nstates, 1, statesize], dtype=tf.float32))
    tflitemodel.invoke()
    hyp = tflitemodel.get_tensor(output_details[0]["index"])

    print("".join([chr(u) for u in hyp]))


if __name__ == "__main__":
    fire.Fire(main)
