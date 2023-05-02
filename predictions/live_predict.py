import base64
import cv2
import mediapipe as mp
import time
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
from predictions.predict_landmarks import Prediction
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras_facenet import FaceNet
import pickle
import cv2
import cv2 as cv
from mtcnn.mtcnn import MTCNN

import mediapipe as mp
import numpy as np
from database.data_base_handler import database_handler

class LivePredict:
    """
               This class shall be used for live predictions of landmarks of data on webcam stream for tesing.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

               """

    def __init__(self, mode=None):
        self.prediction = Prediction()
        self.haarcascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        self.detector = MTCNN()
        self.mode=mode
        self.source = 0
        # self.cap = cv2.VideoCapture(self.source)
        # self.success, self.img = self.cap.read()
        # self.cap.release()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.final_name = ""
        self.pose_predicted = 'Detecting'
        self.x, self.y, self.w, self.h=10, 10, 10, 10
        self.mp_pose = mp.solutions.pose
        self.pose =  self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)
        self.time_gap = 7
        self.db_handler = database_handler()

    def get_available_cameras(self):
        available_sources = 0
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_sources = i
                cap.release()
        return available_sources

    def live_predict_pose(self, image):
        """
                Method Name: predict
                Description: This method predict the class from landmark data.
                Output: None
                On Failure: Raise Exception

                Written By: Google Mediapipe, Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
        self.time_gap = 3
        flag = False
        current_time = int(time.time())
        image.flags.writeable = False
        results = self.pose.process(image)
    
        # Draw the pose annotation on the image.
        image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
    
        # Run predictions every 3 seconds
        if (current_time % self.time_gap == 0 and not flag):
            if (results.pose_landmarks is not None):
                    self.pose_predicted = self.prediction.predict(results.pose_landmarks.landmark)
                    print("prediction class:", self.pose_predicted)

        elif current_time % self.time_gap != 0:
            flag = False
        image = cv2.flip(image, 1)
        cv2.putText(image, self.pose_predicted, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        
        return image



    def live_predict_face(self, image, detection_model):
        flag = False
        
        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            image = image

            current_time = int(time.time())
            if (current_time % self.time_gap == 0 and not flag and image is not None):
                # print(image.shape)
                rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                try:
                    if detection_model == "mtcnn":
                        self.x,self.y,self.w,self.h = self.detector.detect_faces(rgb_img)[0]['box']
                        self.time_gap = 7
                    elif detection_model == "haar":
                        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)
                        for x,y,w,h in faces:
                            self.x,self.y,self.w,self.h=x,y,w,h
                        self.time_gap = 1
                    img = rgb_img[self.y:self.y+self.h, self.x:self.x+self.w]
                    img =  cv.resize(img, (160, 160))
                    img = np.expand_dims(img, axis=0)
                        
                    self.final_name = self.prediction.face_predict(img)
                except IndexError:
                    print("no face")
                    self.final_name = "No Face"
            elif current_time % self.time_gap != 0:
                flag = False
            black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            black_image[:]=(255,255,255)                
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=black_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=black_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    self.mp_drawing.draw_landmarks(
                        image=black_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            else:
                img = np.ones((480, 640, 3), dtype = np.uint8)
                img[:]=(255,255,255)
                return img    
            cv.rectangle(black_image, (self.x,self.y), (self.x+self.w, self.y+self.h), (255,0,255), 3)
            black_image = cv2.flip(black_image, 1)
            cv.putText(black_image, str(self.final_name), (black_image.shape[0] - self.x, self.y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,239,80), 3, cv.LINE_AA)   
            return black_image

    def show_pose(self):
        cap = cv2.VideoCapture(self.get_available_cameras())
        while(True):
            success, image = cap.read()
            black_image = self.live_predict_pose(image)
            print(self.final_name)
            print(self.pose_predicted)
            cv2.imshow('MediaPipe Face Mesh', black_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            if not success:
                print("Ignoring empty camera frame.")
                continue


    def show_face(self, detection_model):
        cap = cv2.VideoCapture(self.get_available_cameras())
        while(True):
            success, image = cap.read()
            black_image = self.live_predict_face(image, detection_model)
            cv2.imshow('MediaPipe Face Mesh', black_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            if not success:
                print("Ignoring empty camera frame.")
                continue

        cap.release()
        cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()

    def show_both(self, detection_model="mtcnn"):
        cap = cv2.VideoCapture(self.get_available_cameras())
        while(True):
            success, image = cap.read()
            black_image = self.live_predict_face(image, detection_model)
            # cv2.imshow('MediaPipe Face Mesh', black_image)
            # time.sleep(1)
            success, image = cap.read()
            black_image = self.live_predict_pose(image)
            # cv2.imshow('MediaPipe Face Mesh', black_image)            
            # time.sleep(1)

            # return self.final_name, self.pose_predicted
            print("*"*50)
            print(self.pose_predicted, self.final_name)
            self.db_handler.df_handle(self.pose_predicted, self.final_name)
        cap.release()
        cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()





    def face_yield(self, img):
        cap = cv2.VideoCapture(self.get_available_cameras())

        while(True):
            success, image = cap.read()
            black_image = self.live_predict_face(img)

            ret, jpeg = cv2.imencode('.jpg', black_image)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            cv2.waitKey(1)
        print(error)
        cap.release()

    def pose_yield(self, img):
        # cap = cv2.VideoCapture(self.get_available_cameras())
        while(True):
            # success, image = cap.read()
            black_image = self.live_predict_pose(img)

            ret, jpeg = cv2.imencode('.jpg', black_image)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            cv2.waitKey(1)
        # cap.release()

    def yield_both(self):
        try:
            cap = cv2.VideoCapture(self.get_available_cameras())

            while(True):
                success, image1 = cap.read()
                black_image1 = self.live_predict_face(image1, "mtcnn")
                success, image2 = cap.read()
                black_image2 = self.live_predict_pose(image2)

                combined_image = cv2.hconcat([black_image1, black_image2])
                ret, jpeg = cv2.imencode('.jpg', combined_image)
                self.db_handler.df_handle(self.pose_predicted, self.final_name)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                cv2.waitKey(1)
        except (AttributeError) as error:
            print(error)
        cap.release()

    def get_pose(self, img):
        # cap = cv2.VideoCapture(self.get_available_cameras())
            # success, image = cap.read()
        time.sleep(0.5)
        black_image = self.live_predict_pose(img)
        _, img_encoded = cv2.imencode('.jpg', black_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        # ret, jpeg = cv2.imencode('.jpg', black_image)
        # image_64_encode = base64.b64encode(black_image)
        return (img_base64)
        # cap.release()