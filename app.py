from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Render the HTML template with the video stream and pose estimation results
    return render_template('index.html')

@app.route('/process_landmarks', methods=['POST'])
def process_landmarks():
    landmarks = request.get_json()

    print(pd.DataFrame.from_dict(landmarks).shape)
    return jsonify({'status': 'Success'})

if __name__ == '__main__':
    app.run()
