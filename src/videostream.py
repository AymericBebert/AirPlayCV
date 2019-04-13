#!/usr/bin/env python3

"""
Threaded tests
Inspired by https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
"""

from threading import Thread

import cv2


class WebcamVideoStream:
    def __init__(self, src=0, threaded=False, options=None):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        self.threaded = threaded
        if options is not None:
            for opt, val in options.items():
                self.stream.set(opt, val)
        self.grabbed, self.frame = self.stream.read()
        self.has_new_image = True

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        if self.threaded:
            # start the thread to read frames from the video stream
            Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            self.grabbed, self.frame = self.stream.read()
            self.has_new_image = True

    def read(self):
        if self.threaded:
            self.has_new_image = False
        else:
            self.grabbed, self.frame = self.stream.read()
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        if not self.threaded:
            self.stream.release()
