# STEP 1: Import the necessary modules
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()

MARGIN = 30  # pixels
FONT_SIZE = 5
FONT_THICKNESS = 5
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
modelpath = os.getenv("MODEL_PATH")
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Convert RGB image to BGR
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS
        )

        # Draw handedness (left or right hand).
        text_x = int(hand_landmarks[0].x * annotated_image.shape[1])
        text_y = int(hand_landmarks[0].y * annotated_image.shape[0]) - MARGIN
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))


# STEP 2: Create a HandLandmarker object.
base_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=modelpath),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

# Open a video capture  (webcam or video file)
cap = cv2.VideoCapture(0)  # 0 is the default webcam

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert the frame to RGB (MediaPipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        landmarker.detect_async(mp_image, timestamp_ms=0)

        # Visualize the results (if available)
        detection_result = landmarker.result()
        if detection_result:
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            cv2.imshow("Live Hand Landmarks", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()