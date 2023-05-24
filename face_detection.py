#!pip install mediapipe
#!pip install PyQt5
#@title Library Imports {display-mode: "form"}

# Please refer to requirements.txt for a full list of all libraries and their versions used in this project.

import numpy as np
import cv2
import mediapipe as mp
import os
from pathlib import Path

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

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#@title Load Age Detection Model {display-mode: "form"}
#For static images - input file path:
def static_image_face_detection(image_files, model, set_images_progress):
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        total = len(image_files)
        print("total: ", total)
        i = 0
        for idx, file in enumerate(image_files):
            i += 1
            set_images_progress(100 * i / total)
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

          # Draw face detections of each face.
            if not results.detections:
                continue
            annotated_image = image.copy()
            idx = 0
            for detection in results.detections:
                print(f'Nose tip:')
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(annotated_image, detection)

                data = detection.location_data.relative_bounding_box
                cropped_image = CropImage(
                    image, data.xmin, data.ymin, data.width, data.height)
                if (not cropped_image.any()):
                    continue
                cv2.imwrite(os.path.join("./Cropped_images", Path(file).stem + str(idx) + '.png'), cropped_image)
                idx += 1
                cv2.imwrite(os.path.join("./Annotated_images", Path(file).stem + str(idx) + '.png'), annotated_image)

  
# For webcam input:
def webcam_face_detection(capture, face_detection, model):
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
    success, image = capture.read()

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

  # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            # crop to detection
            data = detection.location_data.relative_bounding_box
            cropped_image = CropImage(
                image, data.xmin, data.ymin, data.width, data.height)
        # predict age
        # prediction = model.predict(cv2.Canny(cv2.resize(cropped_image, (200, 200)), 50, 75))
        # print(prediction)

    image = cv2.flip(image, 1)
    return image
      
# For video input
def video_face_detection(capture, face_detection, model, set_video_progression):
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                          (640, 400))

    if not capture.isOpened():
            print("Cannot read file")
            exit()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    i = 0
    while True:
        success, image = capture.read()
        if image is None:
            break

        i += 1
        set_video_progression(100 * i / length)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                # crop to detection
                # data = detection.location_data.relative_bounding_box
                # cropped_image = CropImage(
                #     image, data.xmin, data.ymin, data.width, data.height)
                # predict age
                # prediction = model.predict(cv2.Canny(cropped_image, 50, 75))
                # print(prediction)

        image = cv2.flip(image, 1)
        out.write(cv2.resize(image, (640, 400)))
    out.release()
