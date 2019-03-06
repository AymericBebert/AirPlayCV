"""
Cython code to have fast image operations.

To build, go to the project root folder and run:
./src/cyvisord/cython-setup.py build_ext --inplace
"""

cimport cython

import numpy as np
cimport numpy as np


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cdef void mark_black_squares(np.ndarray[np.uint8_t, ndim=2] img, np.uint8_t threshold, int block_radius):
    """Mark all the black squares in the bottom part of the image with a white X"""
    cdef Py_ssize_t i, j, di, dj, dx
    cdef bint passed;

    for i in range(img.shape[0] // 2, img.shape[0] - block_radius - 1):
        for j in range(block_radius, img.shape[1] - block_radius - 1):
            passed = True
            for di in range(-block_radius, block_radius+1):
                for dj in range(-block_radius, block_radius+1):
                    if img[i+di, j+dj] > threshold:
                        passed = False
                        break
                if not passed:
                    break
            if passed:
                for dx in range(-block_radius, block_radius+1):
                    img[i+dx, j+dx] = 255
                    img[i-dx, j+dx] = 255


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cdef void draw_square(np.ndarray[np.uint8_t, ndim=2] img, int x_min, int x_max, int y_min, int y_max, int thickness):
    """Draw the square defined by x in [x_min, x_max[ and y in [y_min, y_max[. Thickness can be chosen"""
    cdef Py_ssize_t i, j, k
    for i in range(y_min, y_max):
        for k in range(thickness):
            img[i, x_min+k] = 255
            img[i, x_max-k-1] = 255
    for j in range(x_min, x_max):
        for k in range(thickness):
            img[y_min+k, j] = 255
            img[y_max-k-1, j] = 255


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cpdef bint contains_black_squares(np.ndarray[np.uint8_t, ndim=2] img,
                                  int x_min, int x_max, int y_min, int y_max,
                                  np.uint8_t threshold, int block_radius):
    """Return True if the specified zone contains at least a square of radius block_radius. Draw the boundaries"""

    cdef Py_ssize_t i, j, di, dj, dx
    cdef bint passed

    for i in range(y_min + block_radius, y_max - block_radius - 1):
        for j in range(x_min + block_radius, x_max - block_radius - 1):
            passed = True
            for di in range(-block_radius, block_radius+1):
                for dj in range(-block_radius, block_radius+1):
                    if img[i+di, j+dj] > threshold:
                        passed = False
                        break
                if not passed:
                    break
            if passed:
                for dx in range(-block_radius, block_radius+1):
                    img[i+dx, j+dx] = 255
                    img[i-dx, j+dx] = 255

                draw_square(img, x_min, x_max, y_min, y_max, 2)  # Bold square
                return True

    draw_square(img, x_min, x_max, y_min, y_max, 1)  # Thin square
    return False
