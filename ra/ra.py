import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.multiprocessing as mp
import copy
from time import sleep
from pathlib import Path
from ra.agent import NNAgent, Transition
from ra.wrappers.tensor import TensorWrapper


class ParetoAscent(NNAgent):

    def __init__(self, env,
                 policy=None,
                 memory=None,
                 actor=None,
                 gamma=1.,
                 lr=1e-3,
                 lambda_=None,
                 **nn_kwargs):
        optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        super(ParetoAscent, self).__init__(optimizer=optimizer, **nn_kwargs)
        # tensor wrapper to make batches of steps, and convert np.arrays to tensors
        env = TensorWrapper(env)
        self.env = env
        self.actor = actor
        self.policy = policy
        self.memory = memory

        self.gamma = gamma
        self.lambda_ = lambda_

        self.mean_return = 0.

    def start(self, log=None):
        obs = self.env.reset()
        return {'observation': obs,
                'terminal': False}

    def step(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'])
            action = self.policy(actor_out, log=log)
        next_obs, reward, terminal, _ = self.env.step(action)
        # if torch.all(next_obs == previous['observation']):
        #     terminal = torch.ones_like(terminal, dtype=bool)
        
        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal)

        self.memory.add(t, log.episode)
        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal}

    def end(self, log=None, writer=None):
        episode = self.memory.get_episode()
        # rewards to discounted returns
        returns = episode.reward.clone()
        for i in range(len(returns)-1, 0, -1):
            returns[i-1] += self.gamma * returns[i]

        # update mean return, use it as baseline
        self.mean_return += (returns[0] - self.mean_return)/log.episode
        returns = returns - self.mean_return
            
        # reinforce loss is NLLLoss
        actor_out = self.actor(episode.observation)
        log_prob = self.policy.log_prob(episode.action, actor_out)
        # self.optimizer_step(loss)
        # compute the gradients for each objective separately
        all_grads = []
        for r_i in range(returns.shape[-1]):
            loss = -log_prob*returns[:, r_i, None]
            # self.optimizer_step(loss)
            self.optimizer.zero_grad()
            loss = loss.mean()
            # retain graph to redo backprop with other objectives
            loss.backward(retain_graph=True)
            # go over each parameter to retrieve gradients
            for i, p in enumerate(self.actor.parameters()):
                grad = p.grad.unsqueeze(0).clone()
                if r_i == 0:
                    all_grads.append(grad)
                else:
                    all_grads[i] = torch.cat((all_grads[i], grad), dim=0)
        # use ascent simplex to update gradients
        for i, p in enumerate(self.actor.parameters()):
            lambda_ = self.lambda_.view(-1, *np.ones(p.grad.dim(), int))
            p.grad = torch.sum(lambda_*all_grads[i], dim=0)
        # optimizer step using modified gradients
        self.optimizer.step()

    def evalstep(self, previous, log=None):
        actor_out = self.actor(previous['observation'])
        action = self.policy(actor_out, log=log)
        next_obs, reward, terminal, info = self.env.step(action)
        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'env_info': info}

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'policy': self.policy,
                'memory': self.memory}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.policy = sd['policy']
        self.memory = sd['memory']


def sample_simplex(n):
        # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
        w_e = torch.distributions.Exponential(1).sample((n,))
        w_e = w_e/w_e.sum() 
        return w_e

from itertools import combinations
def walk(num_dims, samples_per_dim):
    """
    A generator that returns lattice points on an n-simplex.
    """
    max_ = samples_per_dim + num_dims - 1
    for c in combinations(range(max_), num_dims):
        c = list(c)
        yield [(y - x - 1) / (samples_per_dim - 1)
               for x, y in zip([-1] + c, c + [max_])]


class RA(object):

    def __init__(self, 
                 make_env,
                 n_processes=1,
                 **agent_kwargs):

        self.make_env = make_env
        self.n_processes = n_processes
        self.agent_kwargs = agent_kwargs
        n_objectives = len(make_env().reward_space.low)
        
        self.lambdas = torch.tensor([i for i in walk(n_objectives-1, 15-n_objectives)])

    def train_agent(self, agent_kwargs, rank, **train_kwargs):
        env = self.make_env()
        lambda_ = self.lambdas[rank]
        # env.seed(rank)
        # shallow copy of dict to make individual agent logdirs
        # agent_kwargs = agent_kwargs.copy()
        agent_kwargs = copy.deepcopy(agent_kwargs)
        if 'logdir' in agent_kwargs: 
            agent_kwargs['logdir'] = Path(agent_kwargs['logdir']) / f'agent_{rank}'
        worker = ParetoAscent(env, lambda_=lambda_, **agent_kwargs)
        sleep(1)
        worker.train(**train_kwargs)

    def train(self, **kwargs):
        # mp.set_start_method('spawn')
        # make a process per agent, that will learn concurrently
        for i in range(0, self.lambdas.shape[0], self.n_processes):
            processes = []
            for p_i in range(self.n_processes):
                rank = p_i + i
                if rank >= self.lambdas.shape[0]:
                    break
                p = mp.Process(target=self.train_agent, args=(self.agent_kwargs, rank), kwargs=kwargs)
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
