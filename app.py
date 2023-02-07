from flask import Flask, render_template, Response
import cv2
from predictions.live_predict import LivePredict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    live_predict = LivePredict()
    while True:
        live_predict.live_predict_face()
        processed_image = live_predict.image
        print(processed_image)
        ret, jpeg = cv2.imencode('.jpg', processed_image)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    

@app.route('/video_feed')
def video_feed():
    live_predict = LivePredict(mode="webpage")
    return Response(live_predict.face_yield(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
