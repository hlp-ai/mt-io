import tensorflow as tf
import keras

from ocr.crnn.crnn_ocr import build_model
from ocr.crnn.data import DataLoader


if __name__ == "__main__":
    data_loader = DataLoader(r"../../deva.txt")
    train_dataset, validation_dataset = data_loader.load_data()

    epochs = 100
    early_stopping_patience = 4
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    checkpoint_filepath = './deva2/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True, verbose=1)

    model = build_model(280, 32, len(data_loader.char_to_num.get_vocabulary()) + 1)
    model.summary()

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint_callback],
    )