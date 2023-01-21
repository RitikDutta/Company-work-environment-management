import cv2
import mediapipe as mp
import time
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
from predictions.predict_landmarks import Prediction

class LivePredict:
    """
               This class shall be used for live predictions of landmarks of data on webcam stream for tesing.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

               """

    def __init__(self):
        self.prediction = Prediction()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic

            # self.landmarks = landmark_pb2.NormalizedLandmarkList()
        # self.a='a'

    def live_predict(self, predict_type):
        """
                Method Name: predict
                Description: This method predict the class from landmark data.
                Output: None
                On Failure: Raise Exception

                Written By: Google Mediapipe, Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
        i=0
        sets=0
        # For webcam input:
        cap = cv2.VideoCapture(0)
        predicted = ''
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            refine_face_landmarks=True,
        ) as holistic:
          while cap.isOpened():
            success, image = cap.read()
            if not success:
              print("Ignoring empty camera frame.")
              # If loading a video, use 'break' instead of 'continue'.
              continue
            cv2.putText(image, predicted, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_pose_landmarks_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            cv2.setWindowProperty("MediaPipe Holistic", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            i+=1
            if results.face_landmarks is not None and i >= 20:
                if predict_type=="face":
                    predicted = self.prediction.predict_face(results.face_landmarks)
                if predict_type=="pose":
                    predicted = self.prediction.predict(results.pose_landmarks.landmark)
                print(predicted)
                sets+=1
                i=0
            if cv2.waitKey(5) & 0xFF == 27:
              break
        cap.release()
        cv2.destroyAllWindows()