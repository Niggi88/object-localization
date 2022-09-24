import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
from glob import glob
keras = tf.keras


CHARMANDER_DIR = "data/charmander-tight.png"
BULBASAUR_DIR = "data/bulbasaur-tight.png" 
SQUIRTLE_DIR = "data/squirtle-tight.png" 
FOREGROUND_DIRS = [
    CHARMANDER_DIR,
    BULBASAUR_DIR,
    SQUIRTLE_DIR,
]


def get_name_from_dir(dir):
    return dir.split("/")[1].split("-")[0]


def get_random_background(input_shape):
    background_dirs = glob("background/*.jpg")
    assert len(background_dirs) > 0
    background_dir = np.random.choice(background_dirs)
    img = imageio.imread(background_dir)
    src_height, src_width, _ = img.shape
    height, width, _ = input_shape
    randx = np.random.randint(0, src_width - width)
    randy = np.random.randint(0, src_height - height)
    img = img [randy:randy+height, randx:randx+width, :]
    img = img/255
    return img


def merge_images(img_base, img):
    transparency = img[:,:,3]
    transparency = transparency.astype(np.bool_) 
    img = img[:,:,:3]
    img_base[transparency] = img[transparency]
    return img_base


def load_resized(img_dir):
    img = imageio.imread(img_dir)
    height, widht, _ = img.shape
    resize_x = int(np.random.uniform(int(widht/2), int(widht*1.5)))
    resize_y = int(np.random.uniform(int(height/2), int(height*1.5)))
    img = skimage.transform.resize(img, (resize_y, resize_x), preserve_range=True).astype(np.uint8)
    flip_horizontally = np.random.rand()
    if flip_horizontally > 0.5: img = img[:, ::-1]
    flip_vertically = np.random.rand()
    if flip_vertically > 0.5: img = img[::-1, :]
        
    return img
    

def generate_image(input_shape):
    # create base image
    image = get_random_background(input_shape)
    image_height, image_width, _ = image.shape
    
    # load subimage
    img_index = np.random.randint(0, 3)
    img_dir = FOREGROUND_DIRS[img_index]
    charmander = load_resized(img_dir)
    charmander = charmander / 255
    
    # get random coordinates
    box_height, box_width, _ = charmander.shape
    assert image_height-box_height > 0
    assert image_width-box_width
    y = np.random.randint(0, image_height-box_height)
    x = np.random.randint(0, image_width-box_width)
    
    # merge images
    object_exists = np.random.rand() > 0.5
    if object_exists:
        merged = merge_images(image[y:y+box_height, x:x+box_width, :], charmander)
        image[y:y+box_height, x:x+box_width, :] = merged
    
    y, x, box_height, box_width = y/image_height, x/image_width, box_height/image_height, box_width/image_width
    class_target = np.zeros(shape=(3))
    class_target[img_index] = 1
    target = np.array([y, x, box_height, box_width, *class_target, int(object_exists)])
    return image, target
  
    
def image_generator(input_shape=(512, 512, 3), box_shape=(50, 50), batch_size=32, num_batches=50):
    while True:
        for _ in range(num_batches):
            x = np.zeros(shape=(batch_size, *input_shape))
            y = np.zeros(shape=(batch_size, 8))
            for i in range(batch_size):
                image, target = generate_image(input_shape)
                x[i, :, :, :] = image
                y[i, :] = target
            
            yield x, y
            
            
if __name__ == '__main__':
    img, t = generate_image((128, 128, 3))
    print(t)
    object_name = get_name_from_dir(FOREGROUND_DIRS[np.argmax(t[4:7])])
    exists = t[-1]
    if exists > 0.5: plt.title(object_name)
    else: plt.title("NO OBJECT")
    plt.imshow(img)
    plt.show()