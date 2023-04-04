import cv2
import numpy as np
import matplotlib.pyplot as plt


def opening_algorithm(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening

def opening_edges(image):
    opening = opening_algorithm(image.copy())
    edges = cv2.Canny(opening, 100, 200)
    return edges
