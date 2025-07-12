from flask import Flask, request, render_template
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('model.hdf5')

def prepare_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            image = prepare_image(image)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            return render_template('result.html', prediction=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
