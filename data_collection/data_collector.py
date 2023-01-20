import cv2
import mediapipe as mp
import time
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
from data_processing.converter import Converter


class Data_collection:
    """
               This class shall be used for collection for body position landmarks (eg: x,y,z co-ordinates of nose, hands, visibility in frame).

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

               """
    def __init__(self, class_name, collection_type):
        self.collection_type = collection_type
        self.class_name=class_name
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose
        self.path = 'raw data/training/landmarks.csv'
        self.convertor = Converter()
        # self.landmarks = landmark_pb2.NormalizedLandmarkList()


    def holistic_collection(self):
        cap = cv2.VideoCapture(0)
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
            if self.collection_type == "face":
                self.convertor.convert_mp_to_csv(results, "face_landmarks", self.class_name, 'raw_data/training/collection')
            elif self.collection_type=='pose':
                self.convertor.convert_mp_to_csv(results, "pose_landmarks", self.class_name, 'raw_data/training/collection')
                self.convertor.convert_mp_to_csv(results, "left_hand_landmarks", self.class_name, 'raw_data/training/collection')
                self.convertor.convert_mp_to_csv(results, "right_hand_landmarks", self.class_name, 'raw_data/training/collection')

            if cv2.waitKey(5) & 0xFF == 27:
              break
        cap.release()
        cv2.destroyAllWindows()