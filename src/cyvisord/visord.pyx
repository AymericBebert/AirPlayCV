"""
Cython code to have fast image operations.

To build, go to the project root folder and run:
./src/cyvisord/cython-setup.py build_ext --inplace
"""

cimport cython
from cpython cimport array
import array

import numpy as np
cimport numpy as np

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cpdef detect_black_squares(np.uint8_t[:, :] img, np.uint8_t threshold, int block_radius,
                           int min_i=0, int max_i=100000, int min_j=0, int max_j=100000):
    """Return all the zones where surroundings is a black square"""
    cdef int i, j, di, dj
    cdef bint passed

    cdef int i_min = max(min_i, block_radius)
    cdef int j_min = max(min_j, block_radius)
    cdef int i_max = min(max_i, img.shape[0] - block_radius - 1)
    cdef int j_max = min(max_j, img.shape[1] - block_radius - 1)

    cdef array.array marked_i = array.array('i', [])
    cdef array.array marked_j = array.array('i', [])

    i = i_min
    while i < i_max:
        j = j_min
        while j < j_max:
            passed = True
            for di in range(-block_radius, block_radius+1):
                for dj in range(-block_radius, block_radius+1):
                    if img[i+di, j+dj] > threshold:
                        passed = False
                        break
                if not passed:
                    break
            if passed:
                marked_i.append(i)
                marked_j.append(j)
                # Skip a little
                j += block_radius
            j += 1
        i += 1

    return marked_i, marked_j


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cpdef void draw_square_center(np.uint8_t[:, :] img, int x, int y, int radius, int thickness):
    """Draw the square defined by center in (x, y) and radius. Thickness can be chosen"""
    cdef Py_ssize_t i, j, k
    for i in range(y-radius, y+radius+1):
        for k in range(thickness):
            img[i, x-radius+k] = 255
            img[i, x+radius-k] = 255
    for j in range(x-radius, x+radius+1):
        for k in range(thickness):
            img[y-radius+k, j] = 255
            img[y+radius-k, j] = 255


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cpdef void draw_cross_center(np.uint8_t[:, :] img, int x, int y, int radius):
    """Draw a cross defined by center in (x, y) and radius"""
    cdef Py_ssize_t i, j, k
    for k in range(-radius, radius+1):
        img[y+k, x+k] = 255
        img[y+k, x-k] = 255


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
@cython.cdivision(True)     # unsafe division
cpdef bint is_dark_enough(np.uint8_t[:, :] img, int x, int y, int radius, int threshold):
    """Returns True is, in average, the square id darker then the threshold"""
    cdef Py_ssize_t i, j, k
    cdef int s
    for i in range(y-radius, y+radius+1):
        for j in range(x-radius, x+radius+1):
            s += img[i, j]
    return s / (2*radius+1)**2 < threshold


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)   # turn off negative index wrapping for entire function
cdef void mark_black_squares(np.uint8_t[:, :] img, np.uint8_t threshold, int block_radius):
    """Mark all the black squares in the bottom part of the image with a white X"""
    cdef Py_ssize_t i, j, di, dj, dx
    cdef bint passed

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
cdef void draw_square(np.uint8_t[:, :] img, int x_min, int x_max, int y_min, int y_max, int thickness):
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
cpdef bint contains_black_squares(np.uint8_t[:, :] img,
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
