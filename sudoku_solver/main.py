import cv2
import numpy as np
import os

from solver import SudokuSolver
from finder import SudokuFinder
from util import img_show


def write_missing_values(img, coords, completed_values):
    """
    Using the puzzle corner coordinates and the puzzle matrix
     completed values, write the missing values on the image (img).

     Some of these values are not the best, like the font_scale,
     top_offset and left_offset. But these are good enough estimates.
    :param img:
    :param coords:
    :param completed_values:
    :return:
    """
    color_green = (120, 255, 140)
    color_blueish = (255, 80, 70)

    p1, p2, p3, p4 = coords
    poly_points = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p4[0], p4[1]],
                            [p3[0], p3[1]]], dtype=np.int32)
    cv2.polylines(img, [poly_points], isClosed=True, color=color_green,
                  thickness=3)

    puzzle_cell_w = (p2[0] - p1[0]) / 9
    puzzle_cell_h = (p3[1] - p1[1]) / 9
    top_padding = p1[1]
    left_padding = p1[0]
    left_diff = (p3[0] - p1[0]) / 9
    top_diff = (p2[1] - p1[1]) / 9
    font_scale = puzzle_cell_h / 23

    top_offset = int(puzzle_cell_h / 1.2)
    left_offset = int(puzzle_cell_w / 2.8)

    for x, y, digit in completed_values:
        digit = str(digit)
        y1 = left_padding + int(puzzle_cell_w * y) + left_offset
        y1 += int(left_diff * x)
        x1 = top_padding + int(puzzle_cell_h * x) + top_offset
        x1 += int(top_diff * y)

        cv2.putText(img, digit, (y1, x1),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=color_blueish,
                    thickness=1, lineType=cv2.LINE_AA)

    return img


def load_test_image(test_img_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, 'test_images', test_img_name)
    return cv2.imread(file_name, cv2.IMREAD_COLOR)


def main():
    test_img_names = [
        'handwritten.jpg',  # 0, unable to read all digits
        'site_sudoku.png',  # 1, misreads some digits
        'sudoku_test_rotated_ccw.png',  # 2, ok
        'sudoku_test_clear.png',  # 3, ok
        'sudoku_test_rotated_cw.png',  # 4, ok
        'sudoku_test_clear_smaller.png',  # 5, ok
        'sudoku_sample.png'  # 6, ok
    ]
    img = load_test_image(test_img_names[4])

    sf = SudokuFinder(img, debug_mode=True  )
    puzzle, coords = sf.find_puzzle()
    ss = SudokuSolver(puzzle)
    solved = ss.solve(seconds_limit=4)
    if solved:
        completed_values = ss.get_completed_values()
        write_missing_values(img, coords, completed_values)
        img_show(img)
    else:
        print("Could not solve puzzle in under 4 seconds.")


if '__main__' == __name__:
    main()
