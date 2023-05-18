
import os
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input 
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template  
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)



def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = np.argmax(model.predict(x), axis=1)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        print(preds)
        if preds == 0:
            res  = 'Normal'
        if preds == 1:
            res = 'Pneumonia'
        return res
    return None


if __name__ == '__main__':
    app.run(debug=True)

