#!/usr/bin/env python3

import os
import logging

import cv2
import simpleaudio as sa

from src.utils import resolve_path
import src.cyvisord.visord as cyvo


if __name__ == "__main__":
    logging.basicConfig(level='INFO', format='%(asctime)s [%(name)s/%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    cap = cv2.VideoCapture(0)
    cap.set(3, 200)
    cap.set(4, 100)
    # cap.set(15, -8.0)

    samples_folder = resolve_path("samples")

    flam01 = os.path.join(samples_folder, "flam-01.wav")
    flam01_wo = sa.WaveObject.from_wave_file(flam01)

    kick08 = os.path.join(samples_folder, "kick-08.wav")
    kick08_wo = sa.WaveObject.from_wave_file(kick08)

    zones = [
        (130, 140, 100, 110, flam01_wo),
        (120, 130, 130, 140, flam01_wo),
        (110, 120, 160, 170, kick08_wo),
        (120, 130, 190, 200, kick08_wo),
        (130, 140, 220, 230, kick08_wo),
    ]
    n_zones = len(zones)
    found_before = [False] * n_zones
    found = [False] * n_zones

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray, 1)

        # cyvo.mark_black_squares(gray, 24, 2)

        for i in range(n_zones):
            zd = zones[i]
            found[i] = cyvo.contains_black_squares(gray, zd[2], zd[3], zd[0], zd[1], 64, 1)

            if not found[i] and found_before[i]:
                # print(f"Dum!")
                zd[4].play()

            found_before[i] = found[i]

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
