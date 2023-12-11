
<h2 align="center">
<p>Almost State-of-the-art Automatic Speech Recognition in Tensorflow 2</p>
</h2>

## TFLite Convertion

After converting to tflite, the tflite model is like a function that transforms directly from an **audio signal** to **unicode code points**, then we can convert unicode points to string.

1. Install `tf-nightly` using `pip install tf-nightly`
2. Build a model with the same architecture as the trained model _(if model has tflite argument, you must set it to True)_, then load the weights from trained model to the built model
3. Load `TFSpeechFeaturizer` and `TextFeaturizer` to model using function `add_featurizers`
4. Convert model's function to tflite as follows:

```python
func = model.make_tflite_function(**options) # options are the arguments of the function
concrete_func = func.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
```

5. Save the converted tflite model as follows:

```python
if not os.path.exists(os.path.dirname(tflite_path)):
    os.makedirs(os.path.dirname(tflite_path))
with open(tflite_path, "wb") as tflite_out:
    tflite_out.write(tflite_model)
```

5. Then the `.tflite` model is ready to be deployed

## Training & Testing Tutorial

1. Define config YAML file, see the `config.yml` files in the [example folder](./examples) for reference (you can copy and modify values such as parameters, paths, etc.. to match your local machine configuration)
2. Download your corpus (a.k.a datasets) and create a script to generate `transcripts.tsv` files from your corpus (this is general format used in this project because each dataset has different format). For more detail, see [datasets](./tensorflow_asr/datasets/README.md). **Note:** Make sure your data contain only characters in your language, for example, english has `a` to `z` and `'`. **Do not use `cache` if your dataset size is not fit in the RAM**.
3. [Optional] Generate TFRecords to use `tf.data.TFRecordDataset` for better performance by using the script [create_tfrecords.py](./scripts/create_tfrecords.py)
4. Create vocabulary file (characters or subwords/wordpieces) by defining `language.characters`, using the scripts [generate_vocab_subwords.py](./scripts/generate_vocab_subwords.py) or [generate_vocab_sentencepiece.py](./scripts/generate_vocab_sentencepiece.py). There're predefined ones in [vocabularies](./vocabularies)
5. [Optional] Generate metadata file for your dataset by using script [generate_metadata.py](./scripts/generate_metadata.py). This metadata file contains maximum lengths calculated with your `config.yml` and total number of elements in each dataset, for static shape training and precalculated steps per epoch.
6. For training, see `train.py` files in the [example folder](./examples) to see the options
7. For testing, see `test.py` files in the [example folder](./examples) to see the options. **Note:** Testing is currently not supported for TPUs. It will print nothing other than the progress bar in the console, but it will store the predicted transcripts to the file `output.tsv` and then calculate the metrics from that file.

**FYI**: Keras builtin training uses **infinite dataset**, which avoids the potential last partial batch.


