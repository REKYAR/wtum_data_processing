import sys

import imgui
from OpenGL.GL import *
import OpenGL.GL as gl
import glfw
from imgui.integrations.glfw import GlfwRenderer
import imgui_datascience as imgui_ds

import cv2
import mediapipe as mp
import face_detection as fd

def frame_commands(face_detection, mp_drawing, capture):
    image = fd.webcam_face_detection(capture, face_detection)

    io = imgui.get_io()
    if io.key_ctrl and io.keys_down[glfw.KEY_Q]:
        sys.exit(0)

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File"):
            clicked, selected = imgui.menu_item("Quit", "Ctrl+Q")
            if clicked:
                sys.exit(0)
            imgui.end_menu()
        imgui.end_main_menu_bar()

    imgui.begin("A Window!")

    imgui.text("Hello world")

    imgui_ds.imgui_cv.image(image)

    imgui.end()

def render_frame(impl, window, font, face_detection, mp_drawing, capture):
    glfw.poll_events()
    impl.process_inputs()
    imgui.new_frame()

    gl.glClearColor(0.1, 0.1, 0.1, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    if font is not None:
        imgui.push_font(font)
    frame_commands(face_detection, mp_drawing, capture)
    if font is not None:
        imgui.pop_font()

    imgui.render()
    impl.render(imgui.get_draw_data())
    glfw.swap_buffers(window)


def impl_glfw_init():
    width, height = 1600, 900
    window_name = "Age Detection"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


def main():
    # face detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    capture = cv2.VideoCapture(0)

    # imgui setup
    imgui.create_context()
    window = impl_glfw_init()

    impl = GlfwRenderer(window)

    path_to_font = None
    io = imgui.get_io()
    jb = io.fonts.add_font_from_file_ttf(path_to_font, 30) if path_to_font is not None else None
    impl.refresh_font_texture()

    with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
        while not glfw.window_should_close(window):
            render_frame(impl, window, jb, face_detection, mp_drawing, capture)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()