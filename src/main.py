#!/usr/bin/env python3

import os
import logging

import cv2
import simpleaudio as sa
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from src.utils import resolve_path
import src.cyvisord.visord as cyvo


def detect_zones(img):
    """Will detect the best black zones un the image"""

    # Threshold to adapt to ambient luminosity
    threshold = np.average(img) // 2
    print(f"Zones threshold: {threshold}")

    # Detect the center of black squares (only lower half, centered)
    zones_i, zones_j = cyvo.detect_black_squares(img, threshold, 2,
                                                 min_i=img.shape[0] // 2, max_i=img.shape[0] * 9 // 10,
                                                 min_j=img.shape[1] // 6, max_j=img.shape[1] * 5 // 6)

    zone_centers = np.array(list(zip(zones_i, zones_j)), dtype=int)
    if zone_centers.size == 0:
        return []

    # Use clustering algorithm to gather black squares that are in the same zone
    bw = max(1., min(8., estimate_bandwidth(zone_centers)))
    print(f"Zones bandwidth: {bw}")
    ms = MeanShift(bandwidth=bw)
    ms.fit(zone_centers)

    # For each cluster, make the zone out of the cluster center
    zone_centers = [(int(round(cc[0])), int(round(cc[1]))) for cc in ms.cluster_centers_]
    zone_centers.sort(key=lambda x: x[1])

    # Display the zones
    print(f"Zones centers: {zone_centers}")
    for zi, zj in zone_centers:
        cyvo.draw_square_center(img, zj, zi, 3, 1)

    return zone_centers


def detect_zones_canny(img):
    """Will detect the best black zones un the image"""

    canny = cv2.Canny(gray, 100, 200)
    zone_centers = np.argwhere(canny == 255)

    # Use clustering algorithm to gather black squares that are in the same zone
    bw = max(1., min(50., estimate_bandwidth(zone_centers)))
    print(f"Zones bandwidth: {bw}")
    ms = MeanShift(bandwidth=bw)
    ms.fit(zone_centers)

    # For each cluster, make the zone out of the cluster center
    zone_centers = [(int(round(cc[0])), int(round(cc[1]))) for cc in ms.cluster_centers_]
    zone_centers.sort(key=lambda x: x[1])

    # Display the zones
    print(f"Zones centers: {zone_centers}")
    for zi, zj in zone_centers:
        cyvo.draw_square_center(img, zj, zi, 3, 1)

    return zone_centers


if __name__ == "__main__":
    logging.basicConfig(level='INFO', format='%(asctime)s [%(name)s/%(levelname)s] %(message)s', datefmt='%H:%M:%S')

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray, 1)

        # Refresh zones if Z is pressed
        if cv2.waitKey(1) & 0xFF == ord('z'):
            zc = detect_zones_canny(gray)
            zones = [(i, j, sound_samples[k % n_ss]) for k, (i, j) in enumerate(zc)]
            n_zones = len(zones)
            found_before = [False] * n_zones
            found = [False] * n_zones

        # For each zones, check if it is below or above the darkness threshold
        for i in range(n_zones):
            zd = zones[i]
            if cyvo.is_dark_enough(gray, zd[1], zd[0], 2, 48):
                found[i] = True
                cyvo.draw_square_center(gray, zd[1], zd[0], 3, 2)
            else:
                found[i] = False
                cyvo.draw_square_center(gray, zd[1], zd[0], 3, 1)

            if not found[i] and found_before[i]:
                # print(f"Dum!")
                zd[2].play()

            found_before[i] = found[i]

        # Display the resulting frame
        cv2.imshow('frame', gray)

        # Quit if Q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
