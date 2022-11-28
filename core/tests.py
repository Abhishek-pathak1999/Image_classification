import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import Graph, Session

from PIL import Image
from tensorflow.python import keras
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
# from tensorflow.python.keras.backend import set_session
# from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from keras.applications import vgg16
from django.core.files.storage import FileSystemStorage 



model_graph=tf.compat.v1.get_default_graph()
with model_graph.as_default():
    tf_session=Session()
    with tf_session.as_default():
        model = vgg16.VGG16(weights="imagenet")


def index(request):
    if request.method == "POST":
        fileobj=request.FILES['imageFile']

        fs=FileSystemStorage()
        fn=fs.save(fileobj.name, fileobj)
        fn=fs.url(fn)
        testimg='.'+fn
        img=tf.keras.utils.load_img(testimg, target_size=(224,224))
        x=tf.keras.utils.img_to_array(img)
        image_batch = np.expand_dims(x, axis=0)
        processed_image = vgg16.preprocess_input(image_batch.copy())

        with model_graph.as_default():
            # tf_session=Session()
            with tf_session.as_default():
                features=model.predict(processed_image)
                p = decode_predictions(features)

                print(p)
        return render(request, "index.html")
    else:
        return render(request, "index.html")
    
    return render(request, "index.html")