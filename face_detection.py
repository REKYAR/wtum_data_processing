#!pip install mediapipe
#!pip install PyQt5
#@title Library Imports {display-mode: "form"}

# Please refer to requirements.txt for a full list of all libraries and their versions used in this project.

def CropImage(image, xmin, ymin, xwidth, yheight):
    """Function for cropping images
        image - source image
        xmin, ymin - normalized top-left point coordinates for the cropped image
        xwidth, ywidth - normalized width and height for the cropped image"""
    left = int(xmin * image.shape[1])
    top = int(ymin * image.shape[0])
    width = int(xwidth * image.shape[1])
    height = int(yheight * image.shape[0])
    image_cropped = image[top:top+height, left:left+width]
    return image_cropped

import numpy as np
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#@title Load Age Detection Model {display-mode: "form"}
#For static images - input file path:
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)
    
    detection = results.detections[0].location_data.relative_bounding_box
    cropped_image = CropImage(image, detection.xmin, detection.ymin, detection.width, detection.height)
    cv2.imwrite('cropped_image' + str(idx) + '.png', cropped_image)

# For webcam input:
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# with mp_face_detection.FaceDetection(
#     model_selection=0, min_detection_confidence=0.5) as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(image)

#     # Draw the face detection annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.detections:
#       for detection in results.detections:
#         mp_drawing.draw_detection(image, detection)
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()