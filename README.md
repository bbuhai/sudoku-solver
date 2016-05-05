# sudoku-solver
Finds and solves a sudoku puzzle from an image using opencv

Usage
-----
Install opencv and the packages in `requirements.txt`

An example of how to the code here can be found in `main.main()`

**Notes**: 

- it works on some images, mostly with images that have a clear font. The OCR needs improvements.
- the puzzle must be the biggest element/square inside the image.

Improvements
------------
- make it work with a webcam or a video file, to display the solution live.
- make the OCR better (additional training data?)
- implement a faster sudoku solver (with Knuth's dancing links or http://norvig.com/sudoku.html)


Some of the resources I used:
- http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
- http://opencvpython.blogspot.ro/2012/06/sudoku-solver-part-1.html
- https://github.com/goncalopp/simple-ocr-opencv
