from flask import Flask, request
import os
import json
from predict import predict as make_preds

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        payload = request.get_json()["data"]
        os.makedirs("tmp_upload", exist_ok=True)
        json.dump(payload, open("tmp_upload/predict_payload.json", "w"))
        pred_probs = make_preds("tmp_upload/predict_payload.json", 
                             "google-bert/bert-base-cased")
        return pred_probs
    return


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
