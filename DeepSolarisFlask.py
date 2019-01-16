# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#   python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D,MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
import tensorflow as tf


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None



def vgg16_model(trainable=True):
    base_model = VGG16(False, "imagenet")
    train_from_layer = -2
    for layer in base_model.layers[:train_from_layer]:
        layer.trainable = False
        print("{} is not trainable".format(layer.name))
    for layer in base_model.layers[train_from_layer:]:
        #layer.trainable = True
        layer.trainable = False
        print("{} is trainable".format(layer.name))
    last_conv_layer = base_model.get_layer("block5_conv3")
    x = GlobalAveragePooling2D()(last_conv_layer.output)
    #x = Flatten()(last_conv_layer.output)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)        
    predictions = Dense(1, activation="sigmoid")(x)
    return Model(base_model.input, predictions)


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = vgg16_model(False)
    model.load_weights('./vgg16_3t_wmp_wr_aachen__01_0.88.hdf5')
    global graph
    graph = tf.get_default_graph()

    
    
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(75, 75))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(np.array(image))
                #results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                #for (imagenetID, label, prob) in results[0]:
                #r = {"label": label, "probability": float(prob)}
                data["predictions"].append(preds[0].tolist())

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
app.run()
