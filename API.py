import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from flask import Flask , request, jsonify
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
import os
from metrix import  iou, dice_coef

with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
    model = tf.keras.models.load_model(os.path.join("UNet", "model.h5"))

app = Flask(__name__)


def save_result( y_pred, save_path):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)*255.0
    cv2.imwrite(save_path, y_pred)
@app.route('/TestModel', methods=['POST'])
def predict():
    image = request.json['image']
    image_bytes = base64.b64decode(image)
    image = Image.open(BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    #predict image
    image = cv2.resize(np.array(image), (256, 256))
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)
    save_result(y_pred, "../BrainTumorSegmentation/dataset/results/1.png")
    return "Image saved"

if __name__ == '__main__':
    # run the flask app
    app.run(debug=True, port=44372)
