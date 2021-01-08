import cv2
from pandas import np
import streamlit as st


def run_the_app():
    DATA_DIR = 'data'
    path_input_image = '18499925491_e3af00ff02_o.jpg'
    original_image = cv2.imread(path_input_image)
    st.image(original_image, caption='')

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption='')

    edges = cv2.Canny(gray, 50, 150)
    st.image(edges, caption='')

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(edges, kernel, iterations=10)
    st.image(erosion, caption='')


    # lines = cv2.HoughLinesP(image=edges, rho=10, theta=np.pi / 180, threshold=5, minLineLength=10, maxLineGap=10)
    # st.image(lines, caption='')


def main():
    run_the_app()


if __name__ == "__main__":
    main()
