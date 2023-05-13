import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from predictions.live_predict import LivePredict
from predictions.predict_landmarks import Prediction
from data_processing.converter import Converter
from database.database_operations import CassandraCRUD
import pandas as pd
# from flask_socketio import SocketIO, emit
# from camera.camera import VideoCamera
import time
import base64
from data_processing.converter import Converter

application = Flask(__name__)

app = application
app.secret_key = "super secret key"

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

lp = LivePredict()
@app.route('/stream')
def stream():
    my_var_face = session.get('my_var_face', None)
    # print(my_var_face *10) 
    return render_template('stream.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    face_image = converter.convert_json_to_face_image(data)
    # cv2.imwrite('received_image.jpg', face_image)
    # cv2.imshow('Received Image', face_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(face_image.shape)
    # print(face_image)
    try:
        prediction_face = (lp.get_both(face_image))
    except:
        prediction_face = "NONE"

    session['my_var_face'] = prediction_face
    return redirect(url_for('stream'))


@app.route('/_stuff_face', methods = ['GET'])
def stuff_face():
    my_var_face = session.get('my_var_face', None)
    return jsonify(result=my_var_face)





converter = Converter()
pdn = Prediction()
@app.route('/media')
def media():
    # my_var = session.get('my_var', None)
    return render_template('media.html')


@app.route('/mp', methods=['POST'])
def process_landmarks():
  data = request.json
  landmarks = data['landmarks']
  landmarks_dataframe = converter.convert_list_to_dataframe(landmarks)
  prediction = pdn.predict_df(landmarks_dataframe)
  print(prediction)
  session['my_var'] = prediction
  return redirect(url_for('media'))



@app.route('/_stuff', methods = ['GET'])
def stuff():
    my_var = session.get('my_var', None)
    return jsonify(result=my_var)

@app.route('/combined')
def combined():
    my_var = session.get('my_var', None)
    my_var_face = session.get('my_var_face', None)
    print("both: {my_var} + {my_var_face}")
    return render_template('combined.html')




@app.route('/both')
def both():
    return render_template('both.html')


@app.route('/process_image2', methods=['POST'])
def process_image2():
    # time.sleep(3)
    try:
        if time.time()%5 > 4:
            data = request.json
            face_image = converter.convert_json_to_face_image(data) 
            if data['slider_state'] == True:
                detection_model="haar"
            elif data['slider_state'] == False:
                detection_model = "mtcnn"

            prediction = lp.get_both(data['landmarks'], face_image, detection_model=detection_model)
            print(prediction)
            
            print(data['slider_state'])
            session['my_var_both'] = prediction
            return redirect(url_for('both'))
    except TypeError as e:
        print("rest state")
        return redirect(url_for('both'))
    finally:
        return "200"



@app.route('/_stuff_both', methods = ['GET'])
def stuff_both():
    my_var_both = session.get('my_var_both', None)
    return jsonify(result=my_var_both)


@app.route('/get_time')
def get_time():
    # my_var_both = session.get('my_var_both', None)
    my_var_both = time.strftime('%H:%M:%S')

    return {'time': my_var_both}


@app.route('/train')
def train():
    return render_template('train_face.html')


@app.route('/process_train', methods=['POST'])
def process_train():
    files = request.files.getlist('images')
    text = request.form['text']
    for file in files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        print('Image shape:', image.shape)
    print(text)
    return jsonify({'message': 'Images processed successfully.'})





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)