from PIL import Image, UnidentifiedImageError
import pickle as pkl
from io import BytesIO
import requests
from requests.exceptions import InvalidSchema, MissingSchema
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

loaded_model = pkl.load(open('model_cats_dogs.pkl', 'rb'))
labels = loaded_model['class_name']
loaded_model = loaded_model['model']

################################## Loading a Image from a web browser URL #####################################

img_url = "https://static.vecteezy.com/system/resources/thumbnails/002/098/203/small_2x/silver-tabby-cat-sitting-on-green-background-free-photo.jpg"
if img_url:
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        print(response.status_code)

    except MissingSchema as e:
        print("Missing URL schema error:", e)
        print('Try a Different URL: http(s)/...')

    except InvalidSchema as e:
        print("Invalid URL schema error:", e)
        print('Try a Different URL: http(s)/...')

    except UnidentifiedImageError as e:
        print("Invalid URL schema error:", e)
        print('URL not found, Try a Different URL / http(s)/...')

    else:
        image = np.expand_dims(img, axis=0)
        image = image / 255
        prediction = loaded_model.predict(image)

        TH = 0.5
        prediction_index = int(prediction[0][0] > TH)
        print(prediction)
        print(f'This Picture is of a {labels[prediction_index][:-1]} with {round(np.max(prediction), 2) * 100} certainty')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')  # Hide the axes
        plt.show()

################################## Loading a Image from a local directory #####################################

else:
    try:
        image = tf.keras.preprocessing.image.load_img('cats_vs_dogs/dogs/4.jpg')

    except FileNotFoundError as e:
        print('File Not Found at', e)
        print('Ensure that the file path is correct')

    else:
        img = image.resize((228,228))
        image_array = tf.keras.preprocessing.image.img_to_array(img)
        images = np.expand_dims(image_array, axis=0)
        images = images / 255
        prediction = loaded_model.predict(images)

        TH = 0.5
        prediction_index = int(prediction[0][0] > TH)
        print(prediction)
        print(f'This Picture is of a {labels[prediction_index][:-1]} with {round(np.max(prediction), 3) * 100} certainty')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()