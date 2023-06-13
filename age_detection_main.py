import sys
import threading
import time

import imgui
import OpenGL.GL as gl
import glfw
from imgui.integrations.glfw import GlfwRenderer
import imgui_datascience as imgui_ds

from tkinter import filedialog

import cv2
import mediapipe as mp
import face_detection as fd

import tensorflow as tf


class GUIController:
    def __init__(self, model):
        self.model = model

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.capture = cv2.VideoCapture(0)

        # imgui setup
        imgui.create_context()
        self.window = self.impl_glfw_init()

        self.impl = GlfwRenderer(self.window)

        path_to_font = None
        io = imgui.get_io()
        self.font = (
            io.fonts.add_font_from_file_ttf(path_to_font, 30)
            if path_to_font is not None
            else None
        )
        self.impl.refresh_font_texture()

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # buttons
        self.camera_button = True
        self.video_button = False
        self.images_button = False

        # model prediction states for video and images
        self.video_running = 0
        self.video_progress = 0
        self.video_thread = None

        self.images_running = 0
        self.images_progress = 0
        self.video_thread = None

        # a mutex for thread safe state changing
        self.video_runnning_mutex = threading.Lock()
        self.video_progress_mutex = threading.Lock()

        self.images_runnning_mutex = threading.Lock()
        self.images_progress_mutex = threading.Lock()

        self.camera_operational = True
        # try out the camera
        if not self.capture.isOpened():
            self.camera_button = False
            self.video_button = False
            self.images_button = True

            self.camera_operational = False
            return

        success, _ = self.capture.read()
        if not success:
            self.camera_button = False
            self.video_button = False
            self.images_button = True

            self.camera_operational = False
            return

    # main propgram loop
    def loop(self):
        while not glfw.window_should_close(self.window):
            self.render_frame()

        self.impl.shutdown()
        glfw.terminate()

    # defines the gui
    def frame_commands(self):
        io = imgui.get_io()
        if io.key_ctrl and io.keys_down[glfw.KEY_Q]:
            sys.exit(0)

        imgui.set_next_window_position(0, 0, 1, pivot_x=0, pivot_y=0)
        imgui.set_next_window_size(io.display_size.x, io.display_size.y)

        imgui.begin(
            "Age detection",
            0,
            imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_SAVED_SETTINGS
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_SCROLL_WITH_MOUSE,
        )

        if imgui.radio_button("Camera capture", self.camera_button):
            self.camera_button = True
            self.video_button = False
            self.images_button = False

        imgui.same_line()
        if imgui.radio_button("Video file", self.video_button):
            self.camera_button = False
            self.video_button = True
            self.images_button = False

        imgui.same_line()
        if imgui.radio_button("Multiple images", self.images_button):
            self.camera_button = False
            self.video_button = False
            self.images_button = True

        # define the three tabs

        # camera capture tab
        if self.camera_button == True:
            if self.camera_operational == False:
                imgui.text("There was an error during camera access")
            else:
                image = fd.webcam_face_detection(
                    self.capture, self.face_detection, self.model
                )
                imgui_ds.imgui_cv.image(image)

        # video tab
        elif self.video_button == True:
            self.video_runnning_mutex.acquire()
            if self.video_running == 0:
                self.video_runnning_mutex.release()
                if imgui.button("Open file"):
                    file_path = filedialog.askopenfilename(
                        initialdir="Videos", title="Choose a video"
                    )
                    if file_path is not None:
                        try:
                            # start a thread for the model to work
                            self.video_running = 1
                            self.video_thread = threading.Thread(
                                name="Video labeling thread",
                                target=self.label_video,
                                args=(file_path,),
                            )
                            self.video_thread.start()
                        except:
                            print(
                                "Could not open this file. Please check if it's a valid video file"
                            )
                            self.images_running = 0
            else:
                self.video_runnning_mutex.release()
                with self.video_progress_mutex:
                    imgui.text(f"Completed in {self.video_progress}%")

        # image tab
        else:
            self.images_runnning_mutex.acquire()
            if self.images_running == 0:
                self.images_runnning_mutex.release()
                if imgui.button("Open file"):
                    file_paths = filedialog.askopenfilenames(
                        title="Choose multiple images"
                    )
                    if file_paths is not None:
                        try:
                            # start a thread for the model to work
                            self.images_running = 1
                            self.images_thread = threading.Thread(
                                name="Images labeling thread",
                                target=self.label_images,
                                args=(file_paths,),
                            )
                            self.images_thread.start()
                        except:
                            print(
                                "Could not open some of these files. Please check if they are valid image files"
                            )
                            self.images_running = 0
            else:
                self.images_runnning_mutex.release()
                with self.images_progress_mutex:
                    imgui.text(f"Completed in {self.images_progress}%")

        imgui.end()

    def set_video_progress(self, value):
        with self.video_progress_mutex:
            self.video_progress = value

    def set_images_progress(self, value):
        with self.images_progress_mutex:
            self.images_progress = value

    def label_video(self, file_path):
        video_capture = cv2.VideoCapture(file_path)
        fd.video_face_detection(
            video_capture, self.face_detection, self.model, self.set_video_progress
        )

        # clean up
        with self.video_runnning_mutex:
            self.video_running = 0

    def label_images(self, file_paths):
        fd.static_image_face_detection(file_paths, self.model, self.set_images_progress)

        # clean up
        with self.images_runnning_mutex:
            self.images_running = 0

    def render_frame(self):
        glfw.poll_events()
        self.impl.process_inputs()
        imgui.new_frame()

        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self.font is not None:
            imgui.push_font(self.font)
        self.frame_commands()
        if self.font is not None:
            imgui.pop_font()

        imgui.render()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    # create a glfw window
    def impl_glfw_init(self):
        width, height = 800, 800
        window_name = "Age Detection"

        if not glfw.init():
            print("Could not initialize OpenGL context")
            sys.exit(1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)

        window = glfw.create_window(int(width), int(height), window_name, None, None)
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            print("Could not initialize Window")
            sys.exit(1)

        return window


def main():
    # Load the model
    try:
        model = tf.keras.models.load_model("Models\\canny_edges.h5")
    except:
        print("Could not load the keras model. Make sure it's present in './Models/'")

    controller = GUIController(model)

    controller.loop()


if __name__ == "__main__":
    main()
