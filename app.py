
from flask import Flask, render_template,request, request,redirect,session
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.utils import load_img,img_to_array

app = Flask(__name__)

own_model = load_model("models/own_model_keras.h5")
model_resnet50 = load_model("models/resnet50_keras.h5")
model_vgg = load_model("models/vgg16_keras.h5")

dic = {0 : '1', 1 : '2',2 : '5', 3 : '10', 4 : '20', 5 : '50', 6 : '100', 7 : '500', 8: '1000'}

# model = load_model("models/own_model_keras.h5")

own_model.make_predict_function()
model_resnet50.make_predict_function()
model_vgg.make_predict_function()


def own_predict_label(img_path):
	i = load_img(img_path, target_size=(224,224))
	i = img_to_array(i)
	i = i.reshape(1, 224,224,3)
	own_p = own_model.predict(i)
	return dic[np.argmax(own_p[0])]

def model_resnet50_predict_label(img_path):
	i = load_img(img_path, target_size=(224,224))
	i = img_to_array(i)
	i = i.reshape(1, 224,224,3)
	model_resnet50_p = model_resnet50.predict(i)
	return dic[np.argmax(model_resnet50_p[0])]

def model_vgg_predict_label(img_path):
	i = load_img(img_path, target_size=(224,224))
	i = img_to_array(i)
	i = i.reshape(1, 224,224,3)
	model_vgg_p = model_vgg.predict(i)
	return dic[np.argmax(model_vgg_p[0])]

@app.route('/', methods=['GET', 'POST'])
def recognizer():
    return render_template('index.html')


@app.route('/own_model', methods=['GET', 'POST'])
def bank_note_recognizer():
    
    imagefile = request.files['imagefile']
    image_path = "static/"+ imagefile.filename
    
    p_own = own_predict_label(image_path)

    return render_template('index.html',prediction_own = p_own,img_path = image_path)

@app.route('/ResNet50_model', methods=['GET', 'POST'])
def ResNet50_model():
    
    imagefile = request.files['imagefile']
    image_path = "static/"+ imagefile.filename
    
    p_resnet50 = model_resnet50_predict_label(image_path)

    return render_template('index.html',prediction_resnet50 = p_resnet50,img_path = image_path)

@app.route('/VGG16_model', methods=['GET', 'POST'])
def VGG16_model():
    
    imagefile = request.files['imagefile']
    image_path = "static/"+ imagefile.filename
    
    p_vgg16 = model_vgg_predict_label(image_path)

    return render_template('index.html',prediction_vgg16 = p_vgg16,img_path = image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
