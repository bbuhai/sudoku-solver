from copy import deepcopy

import cv2
import numpy as np

from digits import get_trained_knn

from error import PuzzleNotFound
from util import uniformize_points, img_show


class SudokuFinder(object):

    PUZZLE_SIZE = 450
    DIGIT_RESIZE_W = 10
    DIGIT_RESIZE_H = 10

    def __init__(self, img, debug_mode=False):
        self.original_img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.puzzle_matrix = None
        self._puzzle_coords = []
        self.debug_mode = debug_mode

    def find_puzzle(self):
        puzzle_img = self._preprocess()
        self.puzzle_matrix = self._get_puzzle_matrix(puzzle_img)
        return self.puzzle_matrix, self._puzzle_coords

    def _preprocess(self):
        def threshold_img(image):
            kernel = (3, 3)
            blurred = cv2.GaussianBlur(image, kernel, 0)
            thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 5, 2)
            return cv2.bitwise_not(thr)

        thr_img = threshold_img(self.img_gray)
        puzzle = self._find_puzzle_section(thr_img)
        thr_img = threshold_img(puzzle)

        return thr_img

    def _find_puzzle_section(self, thr_img):
        """
        Finds the biggest square in the image (that should be the puzzle)
        After the coordinates of the corners are found (top-lef, top-right,
        bottom-left, bottom-right), the perspective is warped so that
        puzzle_img contains only the puzzle.
        :param thr_img:
        :return:
        """
        img, contours, h = cv2.findContours(thr_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest = []
        max_area = -1
        for i in contours:
            area = cv2.contourArea(i)
            if area > 100:  # subjective value
                perimeter = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        if not len(biggest):
            raise PuzzleNotFound

        p1, p2, p3, p4 = uniformize_points(biggest[0][0], biggest[1][0],
                                           biggest[2][0], biggest[3][0])

        self._puzzle_coords.extend([p1, p2, p3, p4])
        pts1 = np.float32([[p1, p2, p3, p4]])
        pts2 = np.float32([[0, 0],
                           [self.PUZZLE_SIZE - 1, 0],
                           [0, self.PUZZLE_SIZE - 1],
                           [self.PUZZLE_SIZE - 1, self.PUZZLE_SIZE - 1]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        puzzle_img = cv2.warpPerspective(self.img_gray, matrix,
                                         (self.PUZZLE_SIZE, self.PUZZLE_SIZE))

        if max_area > self.PUZZLE_SIZE ** 2:
            # some extra blur for larger images to help with OCR
            puzzle_img = cv2.GaussianBlur(puzzle_img, (5, 5), 0)

        if self.debug_mode:
            color_white = (255, 255, 255)
            poly_points = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p4[0], p4[1]],
                                    [p3[0], p3[1]]], dtype=np.int32)
            cv2.polylines(img, [poly_points], isClosed=True, color=color_white,
                          thickness=3)
            img_show(thr_img)

        return puzzle_img

    def _get_puzzle_matrix(self, puzzle_img):
        """
        Returns a 9x9 matrix with the found digits.

        Puzzles may have different font sizes and because of that multiple iterations
        are required to find the best width / height.

        Another idea is to remove the sudoku grid first and then use the hierarchy
        from findContours to retrieve only the top-level contours.
        (http://answers.opencv.org/question/53293/how-to-remove-line-on-music-sheet/)
        :param img:
        :param original:
        :return:
        """
        img_copy = np.copy(puzzle_img)
        img_color = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        i, contours, _ = cv2.findContours(puzzle_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        erode = cv2.erode(img_copy, kernel)
        img_dilate = cv2.dilate(erode, kernel)

        trained_knn = get_trained_knn()
        sudoku_matrix_final = []
        min_w, max_w = 5, 40
        min_h, max_h = 44, 59
        step = 0
        coords_digits_fun = []
        coords_digits_final = []

        for _ in xrange(33):
            coords_digits = []
            sudoku_matrix = np.empty((9, 9), dtype=np.object)
            step += 1
            i = 0
            for cnt in contours:
                if cv2.contourArea(cnt) < 20:
                    # too little to be a digit
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if min_w < w < max_w and min_h - step < h < max_h - step:
                    a = y / 50  # 50 = self.PUZZLE_SIZE / 9
                    b = x / 50
                    if (a, b) in coords_digits:
                        # two values in the same cell -> bad reading
                        break
                    roi = img_dilate[y:y + h, x:x + w]
                    digit = cv2.resize(roi, (self.DIGIT_RESIZE_W, self.DIGIT_RESIZE_H))
                    num_total_px = self.DIGIT_RESIZE_H * self.DIGIT_RESIZE_W
                    test_data = digit.reshape((-1, num_total_px)).astype(np.float32)
                    _, result, _, _ = trained_knn.findNearest(test_data, k=1)

                    coords_digits.append((a, b))
                    coords_digits_fun.append((x + 3, y + 3, int(result[0, 0]), i))
                    sudoku_matrix[a, b] = int(result[0, 0])

                    i += 1
            if len(coords_digits) > 16:
                # apparently there are no uniquely solvable 16-clue grids
                if not len(sudoku_matrix_final):
                    sudoku_matrix_final = np.copy(sudoku_matrix)
                    coords_digits_final = deepcopy(coords_digits_fun)
                else:
                    len_matrix = len(sudoku_matrix[np.where(sudoku_matrix > 0)])
                    len_final = len(sudoku_matrix_final[np.where(sudoku_matrix_final > 0)])
                    if len_matrix > len_final:
                        coords_digits_final = deepcopy(coords_digits_fun)
                        sudoku_matrix_final = np.copy(sudoku_matrix)

        if self.debug_mode:
            for x, y, digit, idx in coords_digits_final:
                idx = str(idx)
                digit = str(digit)
                cv2.putText(img_color, digit, (x, y),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0, 0, 240),
                            thickness=1, lineType=cv2.LINE_AA)
            img_show(img_color)

        return sudoku_matrix_final
