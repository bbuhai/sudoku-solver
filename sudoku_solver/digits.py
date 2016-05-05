import os
from itertools import izip

import numpy as np
import cv2


_current_dir = os.path.dirname(os.path.abspath(__file__))
_train_img_path_1 = os.path.join(_current_dir, 'data', 'digits.png')
# _train_img_path_2 not used (for now, maybe)
_train_img_path_2 = os.path.join(_current_dir, 'data', 'digits_font.jpg')
_train_data_font = os.path.join(_current_dir, 'data', 'train_data.data')
_train_labels_font = os.path.join(_current_dir, 'data', 'train_labels.data')
_train_data_handwritten = os.path.join(_current_dir, 'data', 'train_data_handwritten.npz')
_train_labels_handwritten = os.path.join(_current_dir, 'data', 'train_labels_handwritten.npz')


def _extract_digits(img, resize_w=10, resize_h=10):
    img_copy = np.copy(img)
    i, contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ordered_digits = np.zeros((45, 100, resize_w * resize_h))

    for contour, hierarchy in izip(contours, h[0]):
        # hierarchy = [Next, Previous, First_Child, Parent]
        if hierarchy[3] != -1:
            # contour has no parent, so it's not top level
            continue
        area = cv2.contourArea(contour)
        if area < 21 or area > 180:
            # too small or too big to be a number
            continue

        x, y, w, h = cv2.boundingRect(contour)
        pos_x = x / 20
        pos_y = y / 20
        digit = cv2.resize(img_copy[y:y + h, x:x + w], (resize_w, resize_h))
        ordered_digits[pos_y][pos_x] = digit.reshape(-1, resize_w * resize_h)

    return ordered_digits.reshape(4500, resize_w * resize_h)


def _build_handwritten_data(save_to_files=False):
    img = cv2.imread(_train_img_path_1, cv2.IMREAD_GRAYSCALE)
    img = img[100:, :]  # remove the "0" values since sudoku doesn't have zeroes
    k = np.arange(1, 10)

    train_data = _extract_digits(img).astype(np.float32)
    train_labels = np.repeat(k, 500).astype(np.float32)

    if save_to_files:
        np.savez('data/train_data_handwritten.npz', train_data=train_data)
        np.savez('data/train_labels_handwritten.npz', train_labels=train_labels)
    return train_data, train_labels


def _get_handwritten_data():
    try:
        with np.load(_train_data_handwritten) as data:
            train_data_hand = data['train_data']
        with np.load(_train_labels_handwritten) as data:
            train_labels_hand = data['train_labels']
    except IOError as e:
        raise IOError("Could not load data: {}".format(e))

    if not len(train_data_hand) or not len(train_labels_hand):
        train_data_hand, train_labels_hand = _build_handwritten_data(True)

    return train_data_hand, train_labels_hand


def get_trained_knn(with_handwritten=True):
    knn = cv2.ml.KNearest_create()

    train_data_hand = []
    train_labels_hand = []
    if with_handwritten:
        train_data_hand, train_labels_hand = _get_handwritten_data()

    train_data_font = np.loadtxt(_train_data_font).astype(np.float32)
    train_labels_font = np.loadtxt(_train_labels_font).astype(np.float32)

    train_data = np.concatenate((train_data_font, train_data_hand), axis=0)
    train_labels = np.concatenate((train_labels_font, train_labels_hand), axis=0)
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    return knn

if '__main__' == __name__:
    get_trained_knn()
