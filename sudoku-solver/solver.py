from time import time
from copy import deepcopy

_beginner = [
    [4, 5, 6, None, None, None, None, 7, 9],
    [None, 2, 3, 6, 5, None, 1, 8, 4],
    [None, None, None, None, 3, None, None, None, None],
    [6, 1, 9, 7, 4, None, None, None, None],
    [2, 4, None, None, None, None, None, 3, 7],
    [None, None, None, None, 8, 2, 4, 1, 6],
    [None, None, None, None, 2, None, None, None, None],
    [1, 6, 4, None, 7, 5, 2, 9, None],
    [3, 8, None, None, None, None, 7, 4, 5]
]

_hard = [
    [9, None, None, None, None, 2, None, None, None],
    [None, 5, None, 8, 6, 9, 7, 1, 4],
    [6, None, None, None, 3, None, None, None, 5],
    [None, 2, 4, None, None, None, None, None, None],
    [None, 9, 6, None, 8, None, 1, None, None],
    [None, None, None, 7, None, 6, None, 4, None],
    [None, None, None, None, 1, 8, 3, None, None],
    [None, None, None, 3, None, None, None, None, 2],
    [None, None, 9, None, None, None, None, 7, 1]
]

# 2-3 seconds to solve this one
_hardest = [
    [None, None, None, None, None, None, None, None, None],
    [None, 5, 7, 2, 4, None, None, None, 9],
    [8, None, None, None, None, 9, 4, 7, None],
    [None, None, 9, None, None, 3, None, None, None],
    [5, None, None, 9, None, None, 1, 2, None],
    [None, None, 3, None, 1, None, 9, None, None],
    [None, 6, None, None, None, None, 2, 5, None],
    [None, None, None, 5, 6, None, None, None, None],
    [None, 7, None, None, None, None, None, None, 6],
]


class SudokuSolver(object):
    """
    Solves a sudoku puzzle represented as a 9x9 matrix.

    It solves a sudoku puzzle using recursion.
    A faster solution is required when solving puzzles from a video.

    Other ideas:
    - represent the board as 1x81
    - implement this: http://norvig.com/sudoku.html
    - dancing links
    """
    BOARD_WIDTH = 9
    BOARD_HEIGHT = 9
    VALUES = list(range(1, 10))

    def __init__(self, board):
        if len(board) != self.BOARD_HEIGHT:
            raise ValueError("Not enough rows.")
        for i, row in enumerate(board):
            if len(row) != self.BOARD_HEIGHT:
                raise ValueError("Not enough columns for row num %s." % i)
        self.board = board
        self.original_board = deepcopy(board)
        self.start_time = None
        self.end_time = None
        self.seconds_limit = None

    def solve(self, time_it=False, seconds_limit=None):
        if seconds_limit and seconds_limit < 0:
            raise ValueError("seconds_limit must be > 0")
        self.seconds_limit = seconds_limit
        self.start_time = time()
        solved = self.place_values()
        self.end_time = time()
        if time_it:
            if solved:
                print('Solving took %s seconds.' % (self.end_time - self.start_time))
            else:
                print("Could not solve puzzle in %s seconds." % self.seconds_limit)

        return solved

    def place_values(self):
        if 0 < self.seconds_limit < time() - self.start_time:
            return False
        row, col = self.get_unassigned_coords()
        if row is None or col is None:
            return True

        for k in range(1, 10):
            if self.can_place(k, row, col):
                self.board[row][col] = k
                if self.place_values():
                    return True
                self.board[row][col] = None
            if k == 9:
                return False

        return False

    def get_unassigned_coords(self):
        row, col = None, None
        for i, row in enumerate(self.board):
            for j, v in enumerate(row):
                if self.board[i][j] is None:
                    return i, j
        return row, col

    def check(self):
        for i, row in enumerate(self.board):
            if None in row:
                return False
        return True

    def can_place(self, value, row, col):
        if value is None:
            return False

        if value in self.board[row]:
            return False

        for i in range(self.BOARD_HEIGHT):
            if value == self.board[i][col]:
                return False

        sq_i = row - row % 3
        sq_j = col - col % 3
        for i in range(3):
            for j in range(3):
                if value == self.board[i + sq_i][j + sq_j]:
                    return False

        return True

    def print_board(self):

        for i, row in enumerate(self.board):
            line = ''
            for j, col in enumerate(row):
                if j % 3 == 0:
                    line += ' | '
                line += ' ? ' if col is None else ' %s ' % col
                if j == self.BOARD_WIDTH - 1:
                    line += '|'
            if i % 3 == 0:
                print(' ' + '-' * 36)
            print(line)
        print(' ' + '-' * 36)

    def get_completed_values(self):
        """
        Returns the indices together with the values that the solver added to the board.
        The result looks like this:
        [
            (0, 5, 3),  # board[0][5] = 3
            (1, 2, 7),
        ]
        :return:
        """
        completed_values = []
        for i in xrange(self.BOARD_HEIGHT):
            for j in xrange(self.BOARD_WIDTH):
                try:
                    if self.original_board[i][j] <= 0:
                        completed_values.append((i, j, self.board[i][j]))
                except IndexError:
                    raise ValueError('Invalid matrix. Missing indices ({}, {})'.format(i, ))
        return completed_values


def test_can_place():
    ss = SudokuSolver(_beginner)
    assert ss.can_place(1, 6, 0) is False
    assert ss.can_place(2, 6, 0) is False
    assert ss.can_place(3, 6, 0) is False
    assert ss.can_place(4, 6, 0) is False
    assert ss.can_place(5, 6, 0) is True
    assert ss.can_place(6, 6, 0) is False
    assert ss.can_place(7, 6, 0) is True
    assert ss.can_place(8, 6, 0) is False
    assert ss.can_place(9, 6, 0) is True

if '__main__' == __name__:
    ss = SudokuSolver(_beginner)
    ss.solve(time_it=True, seconds_limit=3)
    ss.print_board()
    print("==" * 20)
    ss = SudokuSolver(_hard)
    ss.solve(time_it=True, seconds_limit=0.2)
    ss.print_board()
    print("==" * 20)
    ss = SudokuSolver(_hardest)
    ss.solve(time_it=True, seconds_limit=3)
    ss.print_board()
