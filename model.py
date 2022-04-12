import tensorflow as tf



def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=128, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=128, input_shape=input_shape, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=1))
    return model
