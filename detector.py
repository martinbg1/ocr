import numpy as np
from preprocessing import load_data_detector
from PIL import Image
from OCR import init_clf, clf_predict
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def sliding_window(image, window_size, stride):
    """
    some boil
    """
    for x in range(0, image.size[0] - window_size[0], stride[0]):
        for y in range(0, image.size[1] - window_size[1], stride[1]):
            yield (x, y, image.crop((
                x, y,
                x + window_size[0],
                y + window_size[1]))
            )


def detect(crop, num_pixels):
    """
    kinda boil
    """
    background_color = 255
    mask = (np.array(crop) < background_color)
    detect_score = np.array(crop)[mask].size / 400.
    return detect_score


def scan(img, window_size, stride, detect_score_threshold):
    """
    boil
    """
    num_pixels = window_size[0]*window_size[1]
    boxes = []
    for roi in sliding_window(img, window_size, stride):
        detect_score = detect(roi[2], num_pixels)
        if detect_score >= detect_score_threshold:
            boxes.append((roi[0], roi[1], detect_score))
    return boxes


def get_iou(box1, box2):
    """
    Hard boil
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    inter = float((x_max-x_min)*(y_max-y_min)
                  ) if x_max > x_min and y_max > y_min else 0
    union = float((box1[2]-box1[0])*(box1[3]-box1[1]) +
                  (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)
    # compute the IoU
    iou = inter/union
    return iou


def nms(boxes, window_size, max_boxes=10, iou_threshold=0.5):
    """
    Even harder boil
    """
    detect_scores = np.array([box[2] for box in boxes])
    boxes_coords = np.array(
        [[box[0], box[1], box[0]+window_size[0],
          box[1]+window_size[1]] for box in boxes])
    nms_indices = []
    # Use get_iou() to get the list of indices corresponding to boxes you keep
    idxs = np.argsort(detect_scores)
    while len(idxs) > 0 and len(nms_indices) < max_boxes:
        last = len(idxs) - 1
        ind_max = idxs[last]
        nms_indices.append(ind_max)
        suppress = [last]
        for i in range(0, last):
            overlap = get_iou(boxes_coords[ind_max], boxes_coords[idxs[i]])
            if overlap > iou_threshold:
                suppress.append(i)
        idxs = np.delete(idxs, suppress)
    boxes = [(boxes_coords[index, 0], boxes_coords[index, 1],
              detect_scores[index]) for index in nms_indices]
    return boxes


def box_to_hog(box, img, window_size):
    """
    Run the box through Histogram of Oriented Gradiant
    """
    x, y = box[0], box[1]
    pil_img = (img.crop((x, y, x + window_size[0], y + window_size[1])))
    hog_img = hog(np.array(pil_img), pixels_per_cell=(
        4, 4), cells_per_block=(2, 2))
    return hog_img


def run(clf, boxes, img, window_size):
    """
    Perfrom necessary preproccesing and classify the boxes
    """
    hog_imgs = []
    for box in boxes:
        hog_img = box_to_hog(box, img, window_size)
        hog_imgs.append(hog_img)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(np.array(hog_imgs))
    y_pred = clf_predict(X_test, clf)
    return y_pred


def plot(image, classified_boxes, window_size):
    """
    Hard boil
    """
    fig1 = plt.figure(dpi=400)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    for box in classified_boxes:
        x_min, y_min, x_max, y_max = box[0][0]-.5, box[0][1]-.5, box[0][0] + \
            window_size[0]-.5, box[0][1]+window_size[1]-.5
        prediction = box[1]
        ax1.text(x_min, y_min-3, "%s" %
                 prediction, color="red", fontsize=3)
        x = [x_max, x_max, x_min, x_min, x_max]
        y = [y_max, y_min, y_min, y_max, y_max]
        line, = ax1.plot(x, y, color="red")
        line.set_linewidth(.5)
    # fig1.savefig("classification.png")
    plt.show()


if __name__ == "__main__":
    # X = load_data_detector()

    # Parameters
    WINDOW_SIZE = (20, 20)
    STRIDE = (1, 1)
    DETECT_SCORE_THRESHOLD = .83
    MAX_BOXES = 100
    IOU_THRESHOLD = .1

    # train the model
    clf = init_clf()

    img = Image.open("./dataset/detection-images/detection-2.jpg")

    # Scan image for boxes
    boxes = scan(img, WINDOW_SIZE, STRIDE, DETECT_SCORE_THRESHOLD)
    # Non-maximum supression
    boxes = nms(boxes, WINDOW_SIZE, MAX_BOXES, IOU_THRESHOLD)
    # classify
    y_pred = run(clf, boxes, img, WINDOW_SIZE)
    # convert form int to chr (0=a, 1=b etc.)
    y_pred_char = [chr(y + 97) for y in y_pred]

    classified_boxes = zip(boxes, y_pred_char)
    # for y in y_pred:
    #     char = chr(y + 97)
    #     y_pred_char.append(char)
    plot(img, classified_boxes, WINDOW_SIZE)
