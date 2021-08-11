import numpy as np
from flask import Flask, request, render_template
from model import KerasClassifier
from keras.models import load_model

# Create flask app
flask_app = Flask(__name__)
a = KerasClassifier()
a.run_tuner()
model = load_model("model.h5")


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.expand_dims(float_features, 0)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The diabetes result is {}".format(np.argmax(prediction[0])))


if __name__ == "__main__":
    flask_app.run(debug=True)
