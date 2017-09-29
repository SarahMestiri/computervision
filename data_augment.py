from keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img
from keras import backend as K
K.set_image_dim_ordering('th')

#the path of images to apply augmentation on them
images_path='train'
#create an instance of ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.2,
        height_shift_range=0.2, featurewise_std_normalization=True)
datagen.flow_from_directory(directory=images_path, target_size=(480,752),color_mode='grayscale', class_mode=None, save_to_dir='saved',save_prefix='keras_')
img = load_img('train/images/photon10.png')
datagen.fit(img)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=32,
                          save_to_dir='saved', save_prefix='tut', save_format='png'):
    i += 1
    if i > 20:
        break
