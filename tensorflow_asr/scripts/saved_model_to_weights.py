import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(prog="Convert saved model to weights")

parser.add_argument("--saved", type=str, default=None, help="Saved model path")

parser.add_argument("output", type=str, default=None, help="output file to store weights")

args = parser.parse_args()

assert args.saved and args.output

model = tf.keras.models.load_model(args.saved)

model.save_weights(args.output)
