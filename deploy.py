from flask import Flask, render_template, request,Response
import pickle
import cv2 as cv
import numpy as np
from flask_cors import CORS, cross_origin
import urllib
import jsonpickle
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import base64
from utils import data_uri_to_cv2_img, value_invert
app = Flask(__name__)

ALPHABET = ["None-لاشيئ","alef-ألف", "beh-باء", "teh-تاء", "theh-ثاء", "jeem-جيم", "hah-حاء", "khah-خاء", "dal-دال", "thal-ذال",
        "reh-راء", "zah-زاى", "seen-سين", "sheen-شين", "sad-صاد", "dad-ضاد", "tah-طاء", "zah-ظاء", "ain-عين",
        "ghain-غين", "feh-فاء", "qaf-قاف", "kaf-كاف", "lam-لام", "meem-ميم", "noon-نون", "heh-هاء", "waw-واو", "yeh-ياء"]

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()
    
@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict():
        # Read the image data from a base64 data URL
        imgstring = request.form.get('data')
        # Convert to OpenCV image
        img = data_uri_to_cv2_img(imgstring)
        print(img.size)
        OCR_model = load_model('save')
        down_width = 32
        down_height = 32
        down_points = (down_width, down_height)
        resized_down = cv.resize(img, down_points, interpolation= cv.INTER_LINEAR)
        resized_down = cv.bitwise_not(resized_down)
        #display("UNKNOWN1.png")
        resized_down = resized_down.reshape(1,32,32,1)
        resized_down = resized_down.astype('float32')
        resized_down = resized_down/255.0
        vec_p = OCR_model.predict(resized_down)
        # determine the label corresponding to vector vec_p
        y_p = np.argmax(vec_p)
        print(y_p)

        return Response(response="{reponse:"+str(ALPHABET[y_p])+", age:"+str(y_p)+"}", status=200, mimetype="application/json")

@app.route('/api/predict', methods=['POST', 'GET'])
def predictApi():
    if request.files['image'].filename != u'':
        file_data = request.files['image'].read()
        nparr = np.fromstring(file_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
        print(img.size)
        cv.imwrite("one.jpg",img)
        OCR_model = load_model('save')
        down_width = 32
        down_height = 32
        down_points = (down_width, down_height)
        resized_down = cv.resize(img, down_points, interpolation= cv.INTER_LINEAR)
        resized_down = cv.bitwise_not(resized_down)
        cv.imwrite("UNKNOWN1.jpg", resized_down)
        #display("UNKNOWN1.png")
        resized_down = resized_down.reshape(1,32,32,1)
        resized_down = resized_down.astype('float32')
        resized_down = resized_down/255.0
        vec_p = OCR_model.predict(resized_down)
        # determine the label corresponding to vector vec_p
        y_p = np.argmax(vec_p)
        print(y_p)

        return Response(response="{reponse:"+str(ALPHABET[y_p])+", age:"+str(y_p)+"}", status=200, mimetype="application/json")
   
if __name__ == '__main__':
    app.run(debug=True)

