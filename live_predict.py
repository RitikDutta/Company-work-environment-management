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
            # self.landmarks = landmark_pb2.NormalizedLandmarkList()
        # self.a='a'

    def live_predict(self):
        """
                Method Name: predict
                Description: This method predict the class from landmark data.
                Output: None
                On Failure: Raise Exception

                Written By: Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
        prediction = Prediction()
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
              # If loading a video, use 'break' instead of 'continue'.
              continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            cv2.putText(image, predicted, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        #     time.sleep(1)
            
            # Draw the pose annotation on the image.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     text_flipped = cv2.flip(image, 1)
        #     image = cv2.flip(image, 1)
            mp_drawing.draw_landmarks(
                    image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
        #     image = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            cv2.setWindowProperty("MediaPipe Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            i+=1
            if (results.pose_landmarks is not None):
                if i >=20:
        #         write_dict_list_to_csv(results.pose_landmarks.landmark, class_name, 'landmarks.csv')
                    predicted = self.prediction.predict(results.pose_landmarks.landmark)
        #             print(results.pose_landmarks)
                    sets+=1
                    print(f"sets collected {sets}")
                    i=0
        #     print(results.pose_landmarks)
            if cv2.waitKey(5) & 0xFF == 27:
              break
        cap.release()
        cv2.destroyWindow('MediaPipe Pose')
    