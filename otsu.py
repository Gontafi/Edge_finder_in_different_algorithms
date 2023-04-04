import cv2
import numpy as np
import matplotlib.pyplot as plt


def otsu_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def otsu_edges(image):
    otsu = otsu_threshold(image.copy())
    edges = cv2.Canny(otsu, 100, 200)
    return edges
