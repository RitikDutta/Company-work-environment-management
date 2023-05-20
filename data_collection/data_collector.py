import cv2
import mediapipe as mp
import time
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
from data_processing.converter import Converter
from mtcnn.mtcnn import MTCNN
import numpy as np


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
        self.path = 'raw_data/training/landmarks.csv'
        self.convertor = Converter()
        self.detector = MTCNN()


        # self.landmarks = landmark_pb2.NormalizedLandmarkList()


    def pose_collection(self):
        """
                Method Name: pose_collection
                Description: This method opens up webcam and collects landmarks co-ordinates data.
                Output: None
                On Failure: Raise Exception

                Written By: Google MediaPipe, Ritik Dutta
                Version: 1.0
                Revisions: None

                        """

        try:
            #get a class name otherwise assign classname as working
            # class_name = input("Enter a Pose for class column")
            # if class_name == '':
            #     class_name = "working"

            #adds a countdown to get user in position for pose collection
            t=5
            print("running pose collection in \n \tpress Esc to close")
            while t > 0:
                    print(t)
                    t -= 1
                    time.sleep(1)

            # For webcam input:
            sets = 0
            cap = cv2.VideoCapture(0)
            with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=2) as pose:
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
                results = pose.process(image)
            #     time.sleep(1)
                
                print(f"sets collected {sets}")
                # Draw the pose annotation on the image.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                cv2.setWindowProperty("MediaPipe Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                print(type(results.pose_landmarks))
                if results.pose_landmarks is not None:
                    self.convertor.convert_mp_to_csv(results.pose_landmarks.landmark, self.class_name, self.path)
                    sets+=1
            #     print(results.pose_landmarks)
                if cv2.waitKey(5) & 0xFF == 27:
                  break
            cap.release()
            cv2.destroyWindow('MediaPipe Pose')

        except Exception as e:
            raise e


    def face_collection(self):
        cam = cv2.VideoCapture(0)
        test_samples = []
        count = 0
        flag = False
        sample = []
        encodings = []
        test_y=[]
        auto_capture = False
        while True:
            # intializing the frame, ret
            ret, frame = cam.read()
            # if statement
            if not ret:
                print('failed to grab frame')
                break
            if auto_capture:    
                current_time = int(time.time())
            else:
                current_time = 1

            if (current_time % 5 == 0 and not flag) or (cv2.waitKey(5) & 0xFF == 32):
                sample = frame
                sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                face = self.detector.detect_faces(sample)
                if len(face) !=0:
                    x,y,w,h = face[0]['box']
                    sample = sample[y:y+h, x:x+w]
                    sample = cv2.resize(sample, (160,160))
                    test_samples.append(sample)
                    encodings.append(self.convertor.get_embedding(sample))
                    test_y.append(self.class_name)
                    
                    count +=1
                    flag = True
                else:
                    print("face not detected")
            elif current_time % 5 != 0:
                flag = False
                
            frame = cv2.putText(frame, f'Images Capured: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('test samples', frame)
            
        #     test_samples.append(frame)
            
            
            k  = cv2.waitKey(1)
            if k%256 == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        print("extraction inportant points from face")
        # face_embeddings = np.load("models/faces_embeddings.npz")
        self.convertor.append_npz_files('models/faces_embeddings.npz', encodings, test_y)

        test = np.load("models/faces_embeddings.npz")
        print("updated labels", test['labels'])



    def face_collection_web(self, image, class_name):
        sample = []
        encodings = []
        test_y=[]
        test_samples = []
        sample = image
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        face = self.detector.detect_faces(sample)
        if len(face) !=0:
            x,y,w,h = face[0]['box']
            sample = sample[y:y+h, x:x+w]
            sample = cv2.resize(sample, (160,160))
            test_samples.append(sample)
            encodings.append(self.convertor.get_embedding(sample))
            test_y.append(class_name)
        else:
            print("face not detected")
        self.convertor.append_npz_files('models/faces_embeddings.npz', encodings, test_y, web=True)

        test = np.load("models/faces_embeddings.npz")
        print("updated labels", test['labels'])