import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from predictions.live_predict import LivePredict
from database.database_operations import CassandraCRUD
import pandas as pd
from flask_socketio import SocketIO, emit
import base64

application = Flask(__name__)

app = application
# socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')

# def gen():
#     live_predict = LivePredict()
#     while True:
#         live_predict.live_predict_face()
#         processed_image = live_predict.image
#         print(processed_image)
#         ret, jpeg = cv2.imencode('.jpg', processed_image)
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


    

# @app.route('/video_feed')
# def video_feed():
#     live_predict = LivePredict(mode="webpage")
#     print("face")
#     return Response(live_predict.face_yield(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/video_feed_pose')
# def video_feed_pose():
#     live_predict2 = LivePredict(mode="webpage")
#     print("pose")
#     return Response(live_predict2.pose_yield(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/db')
def dbo():
    live_predict = LivePredict()
    live_predict.show_both()
    
@app.route("/daily_activity")
def show_tables():
    crud = CassandraCRUD("test_key")
    data = crud.get_db("daily_activity")
    data.set_index(['employee_id'], inplace=True)
    data.index.name=None
    return render_template('daily_activity.html',tables=[data.to_html()],
    titles = ["Daily Activity"])

@app.route("/total_activity")
def show_tables2():
    crud = CassandraCRUD("test_key")
    data = crud.get_db("total_activity")
    data.set_index(['employee_id'], inplace=True)
    data.index.name=None
    return render_template('total_activity.html',tables=[data.to_html()],
    titles = ["Total Activity"])


@app.route('/video_feed_both')
def video_feed_both():
    live_predict3 = LivePredict(mode="webpage")
    return Response(live_predict3.yield_both(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return render_template('video_feed.html')

@app.route('/stream', methods=['GET'])
def web():
        return '''
            <html>
                <body>
                    <h1>Webcam Stream</h1>
                    <video id="video" width="640" height="480" autoplay></video>
                    <canvas id="canvas" width="640" height="480"></canvas>
                    <img id="processed-image" width="640" height="480">
                    <script>
                        const video = document.getElementById('video');
                        const canvas = document.getElementById('canvas');
                        const ctx = canvas.getContext('2d');
                        const processedImage = document.getElementById('processed-image');
                        navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {
                            video.srcObject = stream;
                            video.play();
                        });
                        setInterval(() => {
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            const imgData = canvas.toDataURL('image/jpeg');
                            fetch('/process_image', {
                                method: 'POST',
                                body: JSON.stringify({image: imgData}),
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            }).then((res) => {
                                return res.json();
                            }).then((data) => {
                                processedImage.src = data.image;
                            });
                        }, 1000);
                    </script>
                </body>
            </html>
    '''

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    img_base64 = data['image']
    img = base64.b64decode(img_base64.split(',')[1])
    npimg = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = 50,50,50,50
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    _, img_encoded = cv2.imencode('.jpg', gray)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)