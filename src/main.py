#!/usr/bin/env python3

import os
import sys
import time
import logging
from typing import List, Tuple

import cv2
import simpleaudio as sa
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from src.utils import resolve_path, pretty_duration
import src.cyvisord.visord as cyvo


LOG_FORMAT = "%(asctime)s [%(name)s/%(levelname)s] %(message)s"
LOG_DATE_FMT = "%H:%M:%S"
DARK_THRESHOLD = 92
DEBUG = False


def detect_zones(img: np.ndarray) -> List[Tuple[int, int]]:
    """Will detect the best black zones un the image"""

    # Threshold to adapt to ambient luminosity
    threshold = np.average(img) // 2
    logging.debug(f"Zones threshold: {threshold}")

    # Detect the center of black squares (only lower half, centered)
    zones_i, zones_j = cyvo.detect_black_squares(img, threshold, 2,
                                                 min_i=img.shape[0] // 2, max_i=img.shape[0] * 9 // 10,
                                                 min_j=img.shape[1] // 6, max_j=img.shape[1] * 5 // 6)

    zone_centers = np.array(list(zip(zones_i, zones_j)), dtype=int)
    if zone_centers.size == 0:
        return []

    if DEBUG:
        img2 = img.copy()
        for i in range(zone_centers.shape[0]):
            cyvo.draw_cross_center(img2, zone_centers[i, 1], zone_centers[i, 0], 1)
        cv2.imshow('frame2', img2)

    # Use clustering algorithm to gather black squares that are in the same zone
    bw = max(1., min(8., estimate_bandwidth(zone_centers)))
    logging.debug(f"Zones bandwidth: {bw}")
    ms = MeanShift(bandwidth=bw)
    ms.fit(zone_centers)

    # For each cluster, make the zone out of the cluster center
    zone_centers = [(int(round(cc[0])), int(round(cc[1]))) for cc in ms.cluster_centers_]
    zone_centers.sort(key=lambda x: x[1])

    if DEBUG:
        img2 = img.copy()
        for zc in zone_centers:
            cyvo.draw_cross_center(img2, zc[1], zc[0], 1)
        cv2.imshow('frame3', img2)

    # Display the zones
    logging.debug(f"Zones centers: {zone_centers}")
    for zi, zj in zone_centers:
        cyvo.draw_square_center(img, zj, zi, 3, 1)

    return zone_centers


def detect_zones_canny(img: np.ndarray) -> List[Tuple[int, int]]:
    """Will detect the best black zones un the image"""

    min_i = img.shape[0] // 2
    max_i = img.shape[0] * 9 // 10
    min_j = img.shape[1] // 6
    max_j = img.shape[1] * 5 // 6

    canny = cv2.Canny(img[min_i:max_i, min_j:max_j], 100, 200)
    zone_centers = np.argwhere(canny == 255) + np.array([min_i, min_j])
    # img[min_i:max_i, min_j:max_j] = canny  # TEMP

    if DEBUG:
        img2 = img.copy()
        for i in range(zone_centers.shape[0]):
            cyvo.draw_cross_center(img2, zone_centers[i, 1], zone_centers[i, 0], 1)
        cv2.imshow('frame2', img2)

    # Use clustering algorithm to gather black squares that are in the same zone
    bw = max(1., min(8., estimate_bandwidth(zone_centers)))
    logging.debug(f"Zones bandwidth: {bw}")
    ms = MeanShift(bandwidth=bw)
    ms.fit(zone_centers)

    # For each cluster, make the zone out of the cluster center
    zone_centers = [(int(round(cc[0])), int(round(cc[1]))) for cc in ms.cluster_centers_]
    zone_centers.sort(key=lambda x: x[1])

    if DEBUG:
        img2 = img.copy()
        for zc in zone_centers:
            cyvo.draw_cross_center(img2, zc[1], zc[0], 1)
        cv2.imshow('frame3', img2)

    # Display the zones
    logging.debug(f"Zones centers: {zone_centers}")
    for zi, zj in zone_centers:
        cyvo.draw_square_center(img, zj, zi, 3, 1)

    return zone_centers


def detect_zones_contours(img: np.ndarray, img_bgr: np.ndarray) -> List[Tuple[int, int]]:
    """Use contour detection to find interesting zones"""
    logging.debug("detect_zones_contours start")

    if DEBUG:
        img_bgr = cv2.flip(img_bgr, 1)

    t0 = time.perf_counter()

    colors = {3: (255, 0, 0), 4: (0, 255, 0), 5: (0, 0, 255), 6: (255, 255, 0), 7: (255, 0, 255), 8: (0, 255, 255)}

    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    ret_, thresh = cv2.threshold(img2, 176, 255, cv2.THRESH_BINARY)

    contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    zone_centers = []

    for cnt in contours:
        # Filter out too small or large zones
        area = cv2.contourArea(cnt)
        if area < 64 or area > 1024:
            continue

        # Compute the center of the contour
        mom = cv2.moments(cnt)
        c_x = int(mom["m10"] / mom["m00"])
        c_y = int(mom["m01"] / mom["m00"])

        if c_y < img.shape[0] // 2:
            continue

        zone_centers.append((c_y, c_x))

        if DEBUG:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)  # Use this info?

            logging.debug(f"Contour area : {area}")
            logging.debug(f"Contour sides: {len(approx)}")
            cv2.circle(img_bgr, (c_x, c_y), 3, (153, 102, 255), -1)
            cv2.drawContours(img_bgr, [cnt], -1, colors.get(len(approx), (192, 192, 192)), 2)

    if DEBUG:
        cv2.imshow('frame2', img2)
        cv2.imshow('frame3', thresh)
        cv2.imshow('frame4', img_bgr)

    logging.info(f"detect_zones_contours took {pretty_duration(time.perf_counter() - t0)}")
    return zone_centers


if __name__ == "__main__":
    DEBUG = DEBUG or "debug" in sys.argv

    logging.basicConfig(level=("DEBUG" if DEBUG else "INFO"), format=LOG_FORMAT, datefmt=LOG_DATE_FMT)
    logging.info("Start in DEBUG mode" if DEBUG else "Start")

    # Initialize windows
    cv2.namedWindow("frame")
    cv2.moveWindow("frame", 50, 400)

    if DEBUG:
        cv2.namedWindow("frame2")
        cv2.moveWindow("frame2", 400, 400)
        cv2.namedWindow("frame3")
        cv2.moveWindow("frame3", 750, 400)
        cv2.namedWindow("frame4")
        cv2.moveWindow("frame4", 1100, 400)

    # Initialize the video capture with small dimensions (easier on computation)
    cap = cv2.VideoCapture(0)
    cap.set(3, 200)
    cap.set(4, 100)
    # cap.set(15, -8.0)

    # Load the sound samples
    samples_folder = resolve_path("samples")
    sound_samples = []

    flam01 = os.path.join(samples_folder, "flam-01.wav")
    sound_samples.append(sa.WaveObject.from_wave_file(flam01))

    flam01 = os.path.join(samples_folder, "ophat-05.wav")
    sound_samples.append(sa.WaveObject.from_wave_file(flam01))

    kick08 = os.path.join(samples_folder, "kick-08.wav")
    sound_samples.append(sa.WaveObject.from_wave_file(kick08))

    sdst02 = os.path.join(samples_folder, "sdst-02.wav")
    sound_samples.append(sa.WaveObject.from_wave_file(sdst02))

    n_ss = len(sound_samples)

    # Initializes with 0 zones
    zones = []
    n_zones = len(zones)
    found_before = [False] * n_zones
    found = [False] * n_zones

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # TODO read in grayscale
        # gray = cv2.flip(gray, 1)

        # Key input
        k = cv2.waitKey(1)

        # Quit if Q is pressed
        if k & 0xFF == ord('q'):
            break

        # Refresh zones if Z is pressed
        if k & 0xFF == ord('z'):
            # Detect zones
            # _zc = [(150, 100), (150, 150), (150, 200)]
            # _zc = detect_zones(gray)
            _zc = detect_zones_canny(gray)
            # _zc = detect_zones_contours(gray, frame)

            # Refresh checked zones and sound associated to them
            zones = [(i, j, sound_samples[k % n_ss]) for k, (i, j) in enumerate(_zc)]
            n_zones = len(zones)
            found_before = [False] * n_zones
            found = [False] * n_zones

        # For each zones, check if it is below or above the darkness threshold
        for i in range(n_zones):
            zd = zones[i]
            if cyvo.is_dark_enough(gray, zd[1], zd[0], 2, DARK_THRESHOLD):
                found[i] = True
                cyvo.draw_square_center(gray, zd[1], zd[0], 3, 2)
            else:
                found[i] = False
                cyvo.draw_square_center(gray, zd[1], zd[0], 3, 1)

            if not found[i] and found_before[i]:
                # logging.debug(f"Dum!")
                zd[2].play()

            found_before[i] = found[i]

        # Display the resulting frame
        cv2.imshow('frame', gray)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
