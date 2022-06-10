import numpy as np
import datetime
from dataclasses import dataclass, asdict
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from logger import Logger
from collections import namedtuple


@dataclass
class Log(object):

    total_steps: int
    episode: int
    episode_step: int
    reward: float
    

Transition = namedtuple('Transition', [
    'observation',
    'action',
    'reward',
    'next_observation',
    'terminal'
])


class Agent(object):

    def __init__(self,
                 logdir='runs'):
        self.logdir = logdir
        self.logger = Logger(self.logdir)

    def eval(self, max_steps=np.inf, log=Log(np.inf, 0, 0, 0)):
        e_step = e_reward = 0
        res = self.start(log=Log(log.total_steps, log.episode, e_step, 0))
        all_res = [res]

        # if hasattr(self.env, 'render'):
        #     frames = self.env.render('rgb_array')[None,:]
        # else:
        #     frames = []

        while not res['terminal'] and e_step < max_steps:
            res = self.evalstep(res, log=Log(log.total_steps, log.episode, e_step, 0))
            # if hasattr(self.env, 'render'):
            #     frame = self.env.render('rgb_array')
            #     frames = np.concatenate((frames, frame[None,:]), axis=0)
            e_step += 1
            e_reward += res['reward']
            all_res.append(res)
        # WHC to CHW
        # if frames:
        #     frames = np.expand_dims(np.moveaxis(frames, -1, 1), 0)
        return e_reward, [] #frames

    def train(self, episodes=np.inf, 
                    timesteps=np.inf, 
                    max_steps=np.inf, 
                    eval_freq=np.inf,
                    report=None):
        assert episodes != np.inf or timesteps != np.inf, 'specify either timesteps or episodes for training'

        total_steps = last_eval = e_i = 0
        best_eval_reward = -np.inf
        while e_i < episodes:
            e_i += 1
            e_step = e_reward = 0
            res = self.start(log=Log(total_steps, e_i, e_step, e_reward))
            while not res['terminal'] and e_step < max_steps:
                res = self.step(res, log=Log(total_steps, e_i, e_step, e_reward))
                e_step += 1
                total_steps += 1
                e_reward += res['reward']

            log = Log(total_steps, e_i, e_step, e_reward)
            self.end(log=log)
            print(log)
            # LOGGING
            self.logger.put('train/episode', log.episode, log.total_steps, 'scalar')
            for r_i in range(log.reward.shape[-1]):
                self.logger.put(f'train/reward/{r_i}', log.reward[0, r_i], log.total_steps, 'scalar')
            self.logger.put('train/episode_steps', log.episode_step, log.total_steps, 'scalar')
            # check depending on timesteps or episodes according to user specification
            tot = timesteps if episodes == np.inf else episodes
            cur = total_steps if episodes == np.inf else e_i
            # check if the current fraction of executed steps is bigger than last eval fraction
            if (last_eval+1)*tot*eval_freq/cur <= 1:
                # update last eval fraction
                last_eval += 1
                # do 10 evaluation runs, keep track of best run for video-recording
                eval_reward = []
                eval_best = -np.inf
                eval_frames = None
                for _ in range(10):
                    eval_r, frames = self.eval(max_steps=max_steps, log=log)
                    # if eval_r > eval_best:
                    #     eval_best = eval_r
                    #     eval_frames = frames
                    eval_reward.append(eval_r.numpy())
                eval_reward = np.mean(eval_reward)
                # save model if it's better on average than last time
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    f = Path(self.logdir) / 'checkpoints' / 'agent_best.pt'
                    f.parents[0].mkdir(parents=True, exist_ok=True)
                    torch.save(self.state_dict(), f)

                # LOGGING
                # self.logger.put('eval/reward', eval_best, log.total_steps)
                # for f_i, f in enumerate(eval_frames):
                #     self.logger.put('eval/obs_frame', f, log.total_steps+f_i)

                if report is not None:
                    # assume it's a multiprocessing.Queue
                    report.put((last_eval, eval_reward))

            if total_steps >= timesteps:
                break

    def start(self, log=None):
        raise NotImplementedError()

    def step(self, params, log=None):
        raise NotImplementedError()

    def end(self, log=None):
        pass

    def evalstep(self, params, log=None):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, sd):
        raise NotImplementedError()


class NNAgent(Agent):

    def __init__(self,
                 optimizer=None,
                 clamp_loss=None,
                 clip_grad_norm=None,
                 scheduler=None,
                 scheduler_steps=None,
                 **kwargs):

        super(NNAgent, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.clamp_loss = clamp_loss
        self.clip_grad_norm = clip_grad_norm

        schedulers = {'linear': lambda e: np.amax([(scheduler_steps-e)/scheduler_steps, 0.])}
        if scheduler is not None:
            assert scheduler_steps is not None, 'scheduler should have timesteps'
            scheduler = schedulers[scheduler]
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler)
        self.scheduler = scheduler

    def optimizer_step(self, loss):

        if self.clamp_loss is not None:
            loss = torch.clamp(loss, min=-self.clamp_loss, max=self.clamp_loss)
        
        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()

        if self.clip_grad_norm is not None:
            # get model params directly from optimizer
            for pg in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(pg['params'], self.clip_grad_norm)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        