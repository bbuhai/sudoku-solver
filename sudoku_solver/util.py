from functools import wraps
import os

import cv2
import numpy as np


def img_show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reshape_and_show(arr, w=20, h=20):
    arr = arr.reshape(w, h).astype(np.uint8)
    img_show(arr)


def log_call(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        args_str = ", ".join(map(str, args[1:]))
        print("%s(%s) => %s" % (f.__name__, args_str, r))
        return r
    return wrapper


def img_write(img, name, path=None):
    if not path:
        path = os.path.join('test_images/', name)
    else:
        path = os.path.join(path, name)
    ret = cv2.imwrite(path, img)
    if not ret:
        raise IOError("Could not write image {} to {}".format(name, path))


def uniformize_points(p1, p2, p3, p4):
    """
    Orders 4 points so their order will be top-left, top-right,
    bottom-left, bottom-right.
    A point is a list/tuple made of two values.

    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :return:
    """
    pts = [p1, p2, p3, p4]
    pts.sort(key=lambda x: x[0] + x[1])
    if pts[1][0] < pts[2][0]:
        pts[1], pts[2] = pts[2], pts[1]

    return pts


def _process_wrapper(queue, *args, **kwargs):
    func = kwargs.pop('func')
    r = func(*args, **kwargs)
    queue.put(r)
    queue.close()
