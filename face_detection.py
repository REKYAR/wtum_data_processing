#!pip install mediapipe
#!pip install PyQt5
# @title Library Imports {display-mode: "form"}

# Please refer to requirements.txt for a full list of all libraries and their versions used in this project.

import numpy as np
import cv2
import mediapipe as mp
import os
from pathlib import Path
import math

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
annotation_color = (255, 255, 255)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
padding = 5


def CropImage(image, xmin, ymin, xwidth, yheight):
    """Function for cropping images
    image - source image
    xmin, ymin - normalized top-left point coordinates for the cropped image
    xwidth, ywidth - normalized width and height for the cropped image"""
    left = int(xmin * image.shape[1])
    top = int(ymin * image.shape[0])
    width = int(xwidth * image.shape[1])
    height = int(yheight * image.shape[0])
    image_cropped = image[top : top + height, left : left + width]
    return image_cropped


def DrawRectangle(image, xmin, ymin, xwidth, yheight):
    left = int(xmin * image.shape[1])
    top = int(ymin * image.shape[0])
    width = int(xwidth * image.shape[1])
    height = int(yheight * image.shape[0])
    p1 = (left, top)
    p2 = (left + width, top + height)
    image = cv2.rectangle(image, p1, p2, annotation_color, thickness)


def AddAgeAnnotation(image, xmin, ymin, xwidth, yheight, age):
    left = int(xmin * image.shape[1])
    top = int(ymin * image.shape[0])
    width = int(xwidth * image.shape[1])
    height = int(yheight * image.shape[0])

    age_string = "Age: " + str(age)
    (text_width, text_height), _ = cv2.getTextSize(
        age_string, font, fontScale, thickness
    )

    left_offset = int((width - text_width) / 2)
    org = (left + left_offset, top + height + text_height + padding)
    image = cv2.putText(
        image,
        age_string,
        org,
        font,
        fontScale,
        annotation_color,
        thickness,
        cv2.LINE_AA,
    )


def FilterImage(image):
    # filtered_image = cv2.Canny(cv2.resize(image, (200, 200)), 50, 75)
    filtered_image = cv2.cvtColor(cv2.resize(image, (200, 200)), cv2.COLOR_BGR2GRAY)
    return filtered_image


def ProcessDetection(detection, original_frame, model):
    if not detection:
        return
    # mp_drawing.draw_detection(image, detection)
    # fit image for detection
    data = detection.location_data.relative_bounding_box
    cropped_image = CropImage(
        original_frame, data.xmin, data.ymin, data.width, data.height
    )
    filtered_image = FilterImage(cropped_image)
    # add dimensions so the image fits the model input
    filtered_image = filtered_image[..., np.newaxis]
    filtered_image = np.expand_dims(filtered_image, axis=0)
    # print(filtered_image.shape)
    # predict age
    prediction = model.predict(filtered_image, verbose=0)
    predicted_age = 0
    if int(prediction[0][0]) > 0:
        predicted_age = int(math.sqrt(int(prediction[0][0])))
    # print(prediction)
    DrawRectangle(original_frame, data.xmin, data.ymin, data.width, data.height)
    AddAgeAnnotation(
        original_frame, data.xmin, data.ymin, data.width, data.height, predicted_age
    )


# @title Load Age Detection Model {display-mode: "form"}
# For static images - input file path:
def static_image_face_detection(image_files, model, set_images_progress):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        total = len(image_files)
        print("total: ", total)
        i = 0
        for idx, file in enumerate(image_files):
            i += 1
            set_images_progress(100 * i / total)
            with open(file, "rb") as stream:
                bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            annotated_image = image.copy()
            idx = 0
            for detection in results.detections:
                try:
                    ProcessDetection(detection, annotated_image, model)
                except:
                    continue

            if not os.path.isdir("./Annotated_images"):
                os.mkdir("./Annotated_images")

            cv2.imwrite(
                os.path.join("./Annotated_images", Path(file).stem + str(idx) + ".png"),
                annotated_image,
            )


# For webcam input:
def webcam_face_detection(capture, face_detection, model):
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
    success, captured_frames = capture.read()

    if not success:
        print("Bad frame read")
        exit()

    captured_frames = cv2.flip(captured_frames, 1)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    captured_frames.flags.writeable = False
    if captured_frames.ndim == 3:
        captured_frames = captured_frames[:, :, 0]

    captured_frames = captured_frames[..., np.newaxis]
    captured_frames = captured_frames[..., np.newaxis]

    captured_frames = cv2.cvtColor(captured_frames, cv2.COLOR_RGB2BGR)
    results = face_detection.process(captured_frames)

    # Draw the face detection annotations on the image.
    captured_frames.flags.writeable = True
    if results.detections:
        for detection in results.detections:
            ProcessDetection(detection, captured_frames, model)

    return captured_frames


# For video input
def video_face_detection(capture, face_detection, model, set_video_progression):
    if not capture.isOpened():
        print("Cannot read file")
        exit()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (frame_width, frame_height))

    i = 0
    while True:
        success, image = capture.read()
        if image is None:
            break

        i += 1
        set_video_progression(100 * i / length)
        image = cv2.flip(image, 1)
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
                ProcessDetection(detection, image, model)

        out.write(cv2.resize(image, (frame_width, frame_height)))

    out.release()
