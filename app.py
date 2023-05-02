import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from predictions.live_predict import LivePredict
from data_processing.converter import Converter
from database.database_operations import CassandraCRUD
import pandas as pd
# from flask_socketio import SocketIO, emit
# from camera.camera import VideoCamera
import time
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

@app.route('/stream')
def stream():
    return render_template('stream.html')

lp = LivePredict()

@app.route('/process_image', methods=['POST'])
def process_image():
    # lp = LivePredict()
    data = request.json
    img_base64 = data['image']
    img = base64.b64decode(img_base64.split(',')[1])
    npimg = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # abc = np.frombuffer(base64.b64decode(next(lp.get_pose(frame))))
    current_time = int(time.time())
    print(frame.shape)
    print("-"*40)
    try:
        if (current_time % 7 == 0):
            abc = (lp.get_pose(frame))
            img_data = base64.b64decode(abc)
            img_np = np.frombuffer(img_data, np.uint8)
            abc = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            x, y, w, h = 110,110,150,150
            gray = cv2.rectangle(abc, (x, y), (x+w, y+h), (0, 255, 0), 2)
            _, img_encoded = cv2.imencode('.jpg', gray)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})
            time.sleep(1)
        else:
            return "200"
    except (RuntimeError, ValueError) as e:
        print(e)
        return "200"

    # print("abc: ", type(abc))
    # print(abc.shape)
    # print("gray: ", type(gray))
    # print(gray.shape)




# @app.route('/camera',methods=['POST'])
# def camera():
#     cap=cv2.VideoCapture(0)
#     while True:
#         ret,img=cap.read()
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         cv2.imwrite("static/cam.png",img)

#         # return render_template("camera.html",result=)
#         time.sleep(0.1)
#         return json.dumps({'status': 'OK', 'result': "static/cam.png"})
#         if cv2.waitKey(0) & 0xFF ==ord('q'):
#             break
#     cap.release()
#     # file="/home/ashish/Downloads/THOUGHT.png"
#     # with open(file,'rb') as file:
#     #     image=base64.encodebytes(file.read())
#     #     print(type(image))
#     # return json.dumps({'status': 'OK', 'user': user, 'pass': password});
#     return json.dumps({'status': 'OK', 'result': "static/cam.png"});

# def gen(camera):
#     while True:
#         data= camera.get_frame()

#         frame=data[0]
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)