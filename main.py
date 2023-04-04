import cv2
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
import matplotlib.pyplot as plt

from canny import canny_edge_detection
from watershed import watershed_edges
from opening import opening_edges
from otsu import otsu_edges


def auto_ground_truth(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    canny_edges = cv2.Canny(blurred_image, 100, 200)
    laplacian_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)

    _, laplacian_binary = cv2.threshold(laplacian_edges, 20, 255, cv2.THRESH_BINARY)

    ground_truth = cv2.bitwise_and(canny_edges, laplacian_binary)
    return ground_truth


def performance_metrics(ground_truth, detected_edges):
    gt_edges = ground_truth.flatten()
    de_edges = detected_edges.flatten()

    precision, recall, f1_score, _ = precision_recall_fscore_support(gt_edges, de_edges, average='binary',
                                                                     pos_label=255)
    mcc = matthews_corrcoef(gt_edges, de_edges)

    return precision, recall, f1_score, mcc


def main():
    image = cv2.imread('cat_self.jpg')
    ground_truth = auto_ground_truth(image)

    canny_edges = canny_edge_detection(image, 100, 200)
    watershed = watershed_edges(image)
    opening = opening_edges(image)
    otsu = otsu_edges(image)

    algorithms = ['Canny', 'Watershed', 'Opening', 'Otsu']
    edge_maps = [canny_edges, watershed, opening, otsu]

    results = []

    for edges in edge_maps:
        precision, recall, f1_score, mcc = performance_metrics(ground_truth, edges)
        results.append([precision, recall, f1_score, mcc])

    metrics_df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F1-score', 'MCC'], index=algorithms)
    print(metrics_df)

    titles = ['Original Image', 'Canny Edge Detection', 'Watershed Edges', 'Opening Edges', 'Otsu Edges']
    images = [image, canny_edges, watershed, opening, otsu]

    for i in range(5):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], cmap='gray')
        plt.title(titles[i]), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':
    main()
