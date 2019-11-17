import numpy as np
from PIL import Image
from ocr import clf_init, clf_predict, clf_svm, clf_keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def crop(image, window_size, x, y):
    return image[y : y + window_size[0], x : x + window_size[1]]


def get_candidate_boxes(image, window_size, stride, threshold):
    boxes = []
    padding = [s // 2 for s in window_size]
    for x in range(padding[1], image.shape[1] - padding[1], stride[1]):
        for y in range(padding[0], image.shape[0] - padding[0], stride[0]):
            if np.mean(crop(image, window_size, x, y) != 255) > threshold:
                boxes.append((x, y))
    return np.asarray(boxes, dtype="float")


def nms(boxes, window_size, threshold):
    choices = []

    x1, y1 = boxes.T
    x2, y2 = (boxes + window_size).T

    indices = np.argsort(y2)

    while len(indices) > 0:
        choice = indices[-1]

        choices.append(choice)

        xx1 = np.maximum(x1[choice], x1[indices[:-1]])
        yy1 = np.maximum(y1[choice], y1[indices[:-1]])
        xx2 = np.minimum(x2[choice], x2[indices[:-1]])
        yy2 = np.minimum(y2[choice], y2[indices[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / np.prod(window_size)

        indices = np.delete(
            indices,
            np.concatenate(([len(indices) - 1], np.where(overlap > threshold)[0])),
        )

    boxes[:, 1] -= 3
    return boxes[choices]


if __name__ == "__main__":

    # Parameters
    WINDOW_SIZE = (20, 20)
    STRIDE = (1, 1)
    THRESHOLD = 0.87
    OVERLAP_THRESHOLD = 0.50

    image = np.array(Image.open("./dataset/detection-images/detection-2.jpg"))

    classifier = clf_init(clf_keras)
    boxes = get_candidate_boxes(image, WINDOW_SIZE, STRIDE, THRESHOLD)
    boxes = nms(boxes, WINDOW_SIZE, OVERLAP_THRESHOLD)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)
    for box in boxes:

        # Create a Rectangle patch
        rect = patches.Rectangle(
            box, 20, 20, linewidth=1, edgecolor="r", facecolor="none"
        )

        X = crop(image, WINDOW_SIZE, int(box[0]), int(box[1]))[np.newaxis, ...]
        text = chr(97 + clf_predict(classifier, X))
        ax.text(box[0], box[1], text)
        ax.add_patch(rect)

    plt.show()
