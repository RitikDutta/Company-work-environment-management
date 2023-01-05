from flask import Flask, render_template, request, jsonify
import pandas as pd
from predictions.predict_landmarks import Prediction
prediction = Prediction()
import time
app = Flask(__name__)

@app.route('/')
def index():
    # Render the HTML template with the video stream and pose estimation results
    return render_template('index.html')

@app.route('/process_landmarks', methods=['POST'])
def process_landmarks():
    landmarks = request.get_json()
    # time.sleep(5)
    print(prediction.predict2(request.get_json()))
    return jsonify({'status': 'Success'})

if __name__ == '__main__':
    app.run()
