import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class Cifar10:
    def __init__(self,filename):
        self.filename =filename


    def predictioncifar10(self):
        # load model
        # model = load_model(os.path.join("model", "model_vgg16.h5"))
        model = load_model(os.path.join("model", "cifar_model.h5"))

        imagename = self.filename
        # test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.load_img(imagename, target_size = (32,32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'airplane'
            return [{ "image" : prediction}]
        elif result[0] == 1:
            prediction = 'automobile'
            return [{ "image" : prediction}]
        elif result[0] == 2:
            prediction = 'bird'
            return [{"image":prediction}]
        elif result[0] == 3:
            prediction = 'cat'
            return [{"image":prediction}]
        elif result[0] == 4:
            prediction = 'deer'
            return [{"image":prediction}]
        elif result[0] == 5:
            prediction = 'dog'
            return [{"image":prediction}]
        elif result[0] == 6:
            prediction = 'frog'
            return [{"image":prediction}]
        elif result[0] == 7:
            prediction = 'horse'
            return [{"image":prediction}]
        elif result[0] == 8:
            prediction = 'ship'
            return [{"image":prediction}]
        elif result[0] == 9:
            prediction = 'truck'
            return [{"image":prediction}]



