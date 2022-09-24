import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
keras = tf.keras
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.layers import Concatenate


def custom_loss(y_true, y_pred):
    box_regression_loss = binary_crossentropy(y_true[:, :4], y_pred[:, :4])
    box_loss = box_regression_loss * y_true[:, -1]
    
    class_classification_loss = categorical_crossentropy(y_true[:, 4:7], y_pred[:, 4:7])
    class_loss = class_classification_loss * y_true[:, -1]
    
    exists_loss = binary_crossentropy(y_true[:, -1], y_pred[:, -1])
    return box_loss + class_loss + 0.5 * exists_loss


def get_model(input_shape=(512, 512, 3)) -> Model:
    vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    x = keras.layers.Flatten()(vgg.output)
    x_exists = keras.layers.Dense(1, activation='sigmoid')(x)
    x_box = keras.layers.Dense(4, activation='sigmoid')(x)
    x_class = keras.layers.Dense(3, activation='softmax')(x)
    x = Concatenate()((x_box, x_class, x_exists))
    model = keras.models.Model(vgg.input, x)
    model.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    return model


if __name__ == '__main__':
    model = get_model()
    print(model.output_shape)