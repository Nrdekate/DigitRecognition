# test keras load

# predict
#from scipy.misc import imread
import numpy as np
from keras.models import model_from_json
# test display
import keras
from keras.datasets import mnist
from displaysample import display_sample


(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
from keras import backend as K

if K.image_data_format() == 'channels_first':
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test loss Expected: 0.0309526360580704')
print('Test accuracy:', score[1])
print('Test accuracy Expected: 0.9932')


print('Prediction tests on 5 elements')
for num in range(5):
    #print(train_labels[num])  
    #Print the label converted back to a number
    label = test_labels[num].argmax(axis=0)
    image = test_images[num].reshape([28,28])

    test_image = np.expand_dims(test_images[num], axis=0)
    out = model.predict(test_image).argmax()
    display_sample(image,label, out)
    

print('Wrong predictions wihtin 1000 elements')    
for num in range(1000):
    #print(train_labels[num])  
    #Print the label converted back to a number
    label = test_labels[num].argmax(axis=0)
    image = test_images[num].reshape([28,28])

    test_image = np.expand_dims(test_images[num], axis=0)
    out = model.predict(test_image).argmax()
    if (out != label):
        display_sample(image,label, out)
