#!/usr/bin/env python3

"""
Play sounds, controlled by separated thread with Queue
"""

import threading
from queue import Queue


play_lock = threading.Lock()


class PlaySounds(threading.Thread):
    def __init__(self, sounds):
        threading.Thread.__init__(self, args=(), kwargs={})
        self.sounds = sounds
        self._n_sounds = len(sounds)
        self.queue = Queue()
        self.daemon = True
        self.stopped = False

    def run(self):
        self.stopped = False
        while not self.stopped:
            val = self.queue.get()
            if val is None:  # exit on `None`
                return
            if val == -1:  # do nothing on -1
                continue
            self._play_sound_index(val)

    def add_to_queue(self, val):
        self.queue.put(val)

    def _play_sound_index(self, idx):
        with play_lock:
            self.sounds[idx % self._n_sounds].play()

    def stop(self):
        self.stopped = True
        self.queue.put(None)

    def refresh_sounds(self, new_sounds):
        self.queue.empty()
        self.queue.put(-1)
        self.sounds = new_sounds
        self._n_sounds = len(new_sounds)
