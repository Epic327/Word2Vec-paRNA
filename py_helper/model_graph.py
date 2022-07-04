from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow import expand_dims
import pydot

if __name__ == '__main__':

    model = tf.keras.models.load_model('./model/text2dcnn-k1-s1-m600-7.h5',custom_objects={'expand_dims': expand_dims})
    plot_model(model,to_file='./figure/text2dcnn_model_7.png',show_shapes=True,show_layer_names=False)