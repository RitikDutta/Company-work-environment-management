# import cv2
from flask import Flask, render_template, Response
from predictions.live_predict import LivePredict
from database.database_operations import CassandraCRUD
import pandas as pd

application = Flask(__name__)
app = application

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



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)