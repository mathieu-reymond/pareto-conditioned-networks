import h5py
from pathlib import Path
import numpy as np
import torch
import cv2
import time
import threading
import atexit


def resize_image(frame, max_width=100):
    w, h = frame.shape[:2]
    r = h/w
    w = np.minimum(w, max_width)
    h = int(r*w)
    frame = cv2.resize(frame, (h, w))
    return frame


class Logger(object):

    def __init__(self, logdir='runs/'):
        self.types = {}
        self.to_log = {}
        if logdir is not None:
            self.logdir = Path(logdir)
            self.logdir.mkdir(parents=True, exist_ok=True)
            # create empty log file
            h5py.File(self.logdir / 'log.h5', 'w').close()
        else:
            self.logdir = None
        # writing to file will be done periodically, in a separate thread
        self._write_thread = threading.Thread(target=self.wait_and_write, daemon=True)
        # lock to ensure data won't be tampered with when writing
        self._lock = threading.Lock()
        # at exist, force leftover data to be written
        atexit.register(self.flush)
        self._write_thread.start()

    def wait_and_write(self, wait=30):
        while True:
            time.sleep(wait)
            with self._lock:
                self.flush()

    def flush(self):
        for tag, type_ in self.types.items():
            # if no logdir, don't log
            if self.logdir is None:
                self.to_log[tag] = []
            # if empty skip
            if not self.to_log[tag]:
                continue
            # only open/close during writing
            with h5py.File(self.logdir / 'log.h5', 'r+') as f:
                if type_ == 'scalar':
                    self.log_scalar(tag, f)
                else:
                    self.log_ndarray(tag, f)

    def put(self, tag, value, step, type_):
        if type_ == 'image':
            value = resize_image(value)
        with self._lock:
            if not tag in self.to_log:
                self.types[tag] = type_
                self.to_log[tag] = []
            self.to_log[tag].append((step, value))

    def log_scalar(self, tag, log_file):
        toadd = self.to_log.pop(tag)
        toadd = np.array(toadd)
        self.to_log[tag] = []

        if not tag in log_file:
            log_file.create_dataset(tag, toadd.shape, maxshape=(None, 2), dtype=np.float32)
            log_file[tag].attrs['type'] = self.types[tag]
        else:
            log_file[tag].resize(log_file[tag].len()+len(toadd), 0)

        log_file[tag][-len(toadd):] = np.array(toadd)

    def log_ndarray(self, tag, log_file):
        steps, ndarray = zip(*self.to_log.pop(tag))
        steps, ndarray = np.array(steps), np.stack(ndarray, axis=0)
        self.to_log[tag] = []

        if not (tag+'/ndarray') in log_file:
            log_file.create_dataset(tag + '/step', steps.shape, maxshape=(None,), dtype=np.int32)
            log_file.create_dataset(tag + '/ndarray', ndarray.shape, maxshape=(None,)+tuple(ndarray.shape[1:]), dtype=ndarray.dtype)
            log_file[tag].attrs['type'] = self.types[tag]
        else:
            for t in [tag+'/ndarray', tag+'/step']:
                log_file[t].resize(log_file[t].len()+len(steps), 0)
        log_file[tag+'/step'][-len(steps):] = steps
        log_file[tag+'/ndarray'][-len(ndarray):] = ndarray
        


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    l = Logger('/tmp/')

    env = gym.make('CartPole-v0')
    env.reset(); rew = 0; step = 0; done = False
    while step < 100:
        _, r, done, _ = env.step(env.action_space.sample())
        rew += r
        l.put('reward', rew, step, 'scalar')
        l.put('frame', env.render('rgb_array'), step, 'image')
        # l.log_ndarray('frame')
        step += 1

    # l.log_scalar('reward')
    # l.log_image('frame')

    time.sleep(30)

    log = h5py.File(l.logdir / 'log.h5', 'r')

    plt.figure()
    plt.plot(log['reward'][:,0], log['reward'][:,1])
    plt.show()

    for frame in log['frame/ndarray'][-5:]:
        plt.figure()
        plt.imshow(frame)
        plt.show()

    log.close()