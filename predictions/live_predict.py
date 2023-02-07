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

import mediapipe as mp
import numpy as np

class LivePredict:
    """
               This class shall be used for live predictions of landmarks of data on webcam stream for tesing.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

               """

    def __init__(self):
        self.prediction = Prediction()
            # self.landmarks = landmark_pb2.NormalizedLandmarkList()
        # self.a='a'

    def live_predict(self):
        """
                Method Name: predict
                Description: This method predict the class from landmark data.
                Output: None
                On Failure: Raise Exception

                Written By: Google Mediapipe, Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        i=0
        landmarks1 = landmark_pb2.NormalizedLandmarkList()
        t=2
        print("running pose collection in \n \tpress Esc to close")
        while t > 0:
                print(t)
                t -= 1
                time.sleep(1)
        # For webcam input:
        sets = 0
        cap = cv2.VideoCapture(0)
        predicted = ''
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
          while cap.isOpened():
            success, image = cap.read()
            if not success:
              print("Ignoring empty camera frame.")
              predicted = 'empty'
              continue
            # To improve performance, optionally mark the image as not writeable to
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            cv2.putText(image, predicted, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Draw the pose annotation on the image.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                    image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            cv2.setWindowProperty("MediaPipe Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            i+=1
            if (results.pose_landmarks is not None):
                if i >=20:
                    predicted = self.prediction.predict(results.pose_landmarks.landmark)
                    print("prediction class:", predicted)
                    i=0
        #     print(results.pose_landmarks)
            if cv2.waitKey(5) & 0xFF == 27:
              break
        cap.release()
        cv2.destroyWindow('MediaPipe Pose')
    


    def live_predict_face(self):
        facenet = FaceNet()
        # face_embeddings = np.load("models/faces_embeddings_done_4classes.npz")
        haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        model = pickle.load(open("models/svc.pkl", 'rb'))
        # Y = face_embeddings['labels']
        # encoder = LabelEncoder()
        # encoder.fit(Y)




        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        i=0
        final_name = ""
        d_model = "mtcnn"
        t = 100
        x,y=0,0
        from mtcnn.mtcnn import MTCNN

        detector = MTCNN()

        # For webcam input:
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(0)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                
                success, image = cap.read()
                
                
                if i >=t:
                    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    try:
                        if d_model == "mtcnn":
                            x,y,w,h = detector.detect_faces(rgb_img)[0]['box']
                        elif d_model == "haar":
                            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
                            t = 0.1
                            for x,y,w,h in faces:
                                x,y,w,h=x,y,w,h
                        img = rgb_img[y:y+h, x:x+w]
                        img =  cv.resize(img, (160, 160))
                        img = np.expand_dims(img, axis=0)

                        final_name = self.prediction.face_predict(img)
            #             cv.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 3)
                    except IndexError:
                        print("no face")
                        final_name = "No Face"
                    i=0
                black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                black_image[:]=(255,255,255)
                cv.putText(black_image, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,239,80), 3, cv.LINE_AA)   

                i+=1
                    
                
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #     black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=black_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=black_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=black_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
                    # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Mesh', black_image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()