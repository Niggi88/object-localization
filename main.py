import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
keras = tf.keras
from keras.models import load_model
from keras.models import Model
from model import custom_loss
from keras.utils.generic_utils import get_custom_objects

if tf.test.gpu_device_name():
    print("GPU")
else:
    exit()
    
from data_generator import image_generator, FOREGROUND_DIRS, get_name_from_dir
from model import get_model


def plot(model: Model, input_shape):
    gen = image_generator(input_shape=input_shape, batch_size=1)
    x, y = gen.__next__()
    pred = model.predict(x)
    fig, ax = plt.subplots()
    ax.imshow(x[0])
    
    if y[0, -1] > 0.5:
        class_name = get_name_from_dir(FOREGROUND_DIRS[np.argmax(y[0, 4:7])])
        img_height, img_width, _ = input_shape
        xy = pred[0,1]*img_height-1, pred[0,0]*img_width-1
        width = pred[0,3]*img_width
        height = pred[0,2]*img_height
        rect = patches.Rectangle(xy=xy, width=width, height=height, color="red", fill=None)
        ax.add_patch(rect)
        ax.set_title(class_name)
    else:
        ax.set_title("NO OBJECT")
    plt.show()


def schedule(epoch):
    if epoch > 20:  return 0.000025
    if epoch > 10:  return 0.00005
    return 0.0001


TEST = False
input_shape = (128, 128, 3)



if not TEST:
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    model: Model = get_model(input_shape=input_shape)
    gen = image_generator(input_shape=input_shape, num_batches=50)
    model.fit(gen, epochs=30, steps_per_epoch=30, callbacks=[lr_scheduler])
    model.save("test.model")
# get_custom_objects().update({"custom_loss": custom_loss})
model: Model = load_model("test.model", custom_objects={"custom_loss": custom_loss})
plot(model, input_shape)