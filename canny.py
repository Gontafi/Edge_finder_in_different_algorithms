import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_edge_detection(image, low_threshold, high_threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return canny_edges

# def main():
#     image = cv2.imread('cat_self.jpg')
#
#     canny_edges = canny_edge_detection(image, 100, 200)
#     watershed = watershed_edges(image)
#     opening = opening_edges(image)
#     otsu = otsu_edges(image)
#
#     titles = ['Original Image', 'Canny Edge Detection', 'Watershed Edges', 'Opening Edges', 'Otsu Edges']
#     images = [image, canny_edges, watershed, opening, otsu]
#
#     for i in range(5):
#         plt.subplot(2, 3, i+1), plt.imshow(images[i], cmap='gray')
#         plt.title(titles[i]), plt.xticks([]), plt.yticks([])
#
#     plt.show()

# if __name__ == '__main__':
#     main()