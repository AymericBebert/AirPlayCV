#!/usr/bin/env python3

"""
Main script
"""

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
from src.videostream import WebcamVideoStream
from src.playsounds import PlaySounds
import src.cyvisord.visord as cyvo


# Configuration parameters
LOG_FORMAT = "%(asctime)s [%(name)s/%(levelname)s] %(message)s"
LOG_DATE_FMT = "%H:%M:%S"
DARK_THRESHOLD = 92
DEBUG = False

# Directory where the path to the samples will be based on
SAMPLES_DIRECTORY = resolve_path("samples")

# The ratio width/height of the instrument will be computed, the first match above it will be used
SOUND_NAMES_BY_RATIOS = [
    (0.0, ["whistle_A.wav", "whistle_B.wav", "whistle_C.wav"]),
    (0.5, ["flam-01.wav", "ophat-05.wav", "kick-08.wav", "sdst-02.wav"]),
    (10.0, ["piano_A.wav", "piano_B.wav", "piano_C.wav"]),
]


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
    zone_centers.sort(key=lambda x: -x[1])

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
    """Will detect the best black zones un the image, using canny filter"""

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
    zone_centers.sort(key=lambda x: -x[1])

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


def detect_zones_contours(img: np.ndarray) -> List[Tuple[int, int]]:
    """Use contour detection to find interesting zones"""
    global DARK_THRESHOLD

    logging.debug("detect_zones_contours start")
    start_line = img.shape[0] // 2

    img_bgr = None
    if DEBUG:
        img_bgr = cv2.cvtColor(img[start_line:, :], cv2.COLOR_GRAY2BGR)

    t0 = time.perf_counter()

    colors = {3: (255, 0, 0), 4: (0, 255, 0), 5: (0, 0, 255), 6: (255, 255, 0), 7: (255, 0, 255), 8: (0, 255, 255)}

    img2 = cv2.GaussianBlur(img[start_line:, :], (3, 3), 0)

    # if DEBUG:
    #     cv2.imshow('frame2', img2)
    #     plt.hist(img2.ravel(), 256, [0, 256])
    #     plt.show()

    threshold = np.percentile(img2, 20) // 2  # half luminosity of 20% brighter pixel
    logging.debug(f"Zones threshold: {threshold}")
    ret_, thresh = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)

    contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    zone_centers = []

    for cnt in contours:
        # Filter out too small or large zones
        area = cv2.contourArea(cnt)
        if area < 48 or area > 1024:
            continue

        # Compute the center of the contour
        mom = cv2.moments(cnt)
        c_x = int(mom["m10"] / mom["m00"])
        c_y = int(mom["m01"] / mom["m00"]) + start_line

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
    DARK_THRESHOLD = threshold
    return sorted(zone_centers, key=lambda x: -x[1])


def chose_instrument(zones_ij: List[Tuple[int, int]], play_sound: PlaySounds):
    """Will chose the right instrument according to the zones disposition"""
    min_i = min_j = 1_000_000
    max_i = max_j = 0
    for zi, zj in zones_ij:
        if zi < min_i:
            min_i = zi
        if zi > max_i:
            max_i = zi
        if zj < min_j:
            min_j = zj
        if zj > max_j:
            max_j = zj
    ratio = (max_j - min_j) / (max_i - min_i + 1)
    logging.info(f"Zones x/y ratio: {ratio}")

    chosen_sounds = []
    for r, sns in SOUND_NAMES_BY_RATIOS[::-1]:
        if ratio >= r:
            chosen_sounds = sns
            break

    new_sounds = [sa.WaveObject.from_wave_file(os.path.join(SAMPLES_DIRECTORY, sn)) for sn in chosen_sounds]
    play_sound.refresh_sounds(new_sounds)


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
    webcam_options = {cv2.CAP_PROP_FRAME_WIDTH: 200, cv2.CAP_PROP_FRAME_HEIGHT: 100, cv2.CAP_PROP_FPS: 60}
    vs = WebcamVideoStream(src=0, threaded=True, options=webcam_options).start()

    # Initialize sound player
    ps = PlaySounds([])
    ps.start()

    # Initializes with 0 zones
    zones: List[Tuple[int, int, int]] = []  # for each zone, (i, j, index)
    n_zones: int = 0
    found_before: List[bool] = []
    found: List[bool] = []

    _t0 = _t1 = time.perf_counter()

    while True:
        if not vs.has_new_image:
            time.sleep(0.001)
            continue

        # Capture frame-by-frame
        frame = vs.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.flip(gray, 1)

        # Key input
        k = cv2.waitKey(1) & 0xFF

        # Quit if Q is pressed
        if k == ord('q'):
            break

        # Refresh zones if Z is pressed
        if k == ord('z'):
            # Detect zones
            # _zc = [(150, 100), (150, 150), (150, 200)]  # hardcoded
            # _zc = detect_zones(gray)                    # black zones
            # _zc = detect_zones_canny(gray)              # canny contours
            _zc = detect_zones_contours(gray)           # refined contours

            # Refresh instrument based on zones shape
            chose_instrument(_zc, ps)

            # Refresh checked zones and sound associated to them
            zones = [(i, j, k) for k, (i, j) in enumerate(_zc)]
            n_zones = len(zones)
            found_before = [False] * n_zones
            found = [False] * n_zones

        if DEBUG:
            _t00 = time.perf_counter()
            logging.debug(f"Last check is {pretty_duration(_t00 - _t0)} old ({1/(_t00 - _t0):.2f} fps)")
            _t0 = _t00

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
                ps.add_to_queue(zd[2])

            found_before[i] = found[i]

        if DEBUG:
            _t1 = time.perf_counter()
            logging.debug(f"Square check done in {pretty_duration(_t1 - _t0)}")

        # Display the resulting frame
        cv2.imshow('frame', gray)

    # When everything done, release the capture
    vs.stop()
    cv2.destroyAllWindows()
