
import numpy as np
from flask import Flask,request,render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@flask_app.route("/")

def home():
    return render_template("index.html")
@flask_app.route("/predict",methods=["POST"])
def predict():
    float_feature= [float(x) for x in request.form. values ()]
    features =[np.array(float_feature)]
    prediction = model.predict(features)
    predicted_crop = prediction[0]
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(predicted_crop))
if __name__ == "__main__":
    flask_app.run(debug = True,host='0.0.0.0', port=5000)