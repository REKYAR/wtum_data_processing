import sys

import imgui
import OpenGL.GL as gl
import glfw
from imgui.integrations.glfw import GlfwRenderer
import imgui_datascience as imgui_ds

import cv2
import mediapipe as mp
import face_detection as fd

class GUIController:

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.capture = cv2.VideoCapture(0)

        # imgui setup
        imgui.create_context()
        self.window = self.impl_glfw_init()

        self.impl = GlfwRenderer(self.window)

        path_to_font = None
        io = imgui.get_io()
        self.font = io.fonts.add_font_from_file_ttf(path_to_font, 30) if path_to_font is not None else None
        self.impl.refresh_font_texture()

        self.face_detection =  self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        # buttons
        self.camera_button = True
        self.video_button = False
        self.images_button = False


    def loop(self):
        while not glfw.window_should_close(self.window):
            self.render_frame()

        self.impl.shutdown()
        glfw.terminate()


    def frame_commands(self):
        image = fd.webcam_face_detection(self.capture, self.face_detection)

        io = imgui.get_io()
        if io.key_ctrl and io.keys_down[glfw.KEY_Q]:
            sys.exit(0)

        imgui.set_next_window_position(0, 0, 1, pivot_x = 0, pivot_y = 0)
        imgui.set_next_window_size(io.display_size.x, io.display_size.y)

        imgui.begin( 'Age detection' , 0 , imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE )

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

        if self.camera_button == True:
            imgui_ds.imgui_cv.image(image)
        elif self.video_button == True:
            imgui.text("Video placeholder")
        else:
            imgui.text("Images placeholder")

        imgui.end()


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
    controller = GUIController()

    controller.loop()

if __name__ == "__main__":
    main()