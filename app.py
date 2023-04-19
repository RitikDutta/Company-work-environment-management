from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)

# Route to render the stream.html template
@app.route('/')
def index():
    return render_template('stream.html')

# Route to receive the video stream and process it using OpenCV
@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the base64 encoded image data from the request
    data = request.json['data']

    # Decode the base64 encoded image data and convert it to a numpy array
    image_data = base64.b64decode(data.split(',')[1])
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Read the image array using OpenCV
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)

    # Print the shape of the image to confirm it has been received correctly
    print(image.shape)

    # Process the image using OpenCV
    # ...

    # Return a response
    return 'Video received and processed successfully!'

if __name__ == '__main__':
    app.run()
