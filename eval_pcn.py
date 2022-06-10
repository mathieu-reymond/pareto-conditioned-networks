from main_pcn import DSTModel, WalkroomModel, MinecartModel, SumoModel
from pcn.pcn import Transition, choose_action
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd


device = 'cpu'


def non_dominated(solutions, return_indexes=False):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    if return_indexes:
        return solutions[is_efficient], is_efficient
    else:
        return solutions[is_efficient]


def greedy_action(model, obs, desired_return, desired_horizon):
    log_probs = model(torch.tensor([obs]).to(device),
                      torch.tensor([desired_return]).to(device),
                      torch.tensor([desired_horizon]).unsqueeze(1).to(device))
    log_probs = log_probs.detach().cpu().numpy()[0]
    action = np.argmax(log_probs)
    return action


def run_episode(env, model, desired_return, desired_horizon, max_return):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = greedy_action(model, obs, desired_return, desired_horizon)
        n_obs, reward, done, _ = env.step(action)

        transitions.append(Transition(
            observation=obs,
            action=action,
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            terminal=done
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound, 
        # to avoid negative returns giving impossible desired returns
        # reward = np.array((reward[1], reward[2]))
        desired_return = np.clip(desired_return-reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))
    return transitions


if __name__ == '__main__':
    import argparse
    import uuid
    import os
    import gym
    from gym.wrappers import TimeLimit
    import pathlib
    import h5py

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('model', type=str, help='load model')
    parser.add_argument('--interactive', action='store_true', help='interactive policy selection')
    parser.set_defaults(interactive=False)
    args = parser.parse_args()
    model_dir = pathlib.Path(args.model)
        
    # ===================================
    # LOAD ENVIRONMENT
    # ===================================
    envs = ('dst', 'minecart', 'sumo')
    env = [e for e in envs if e in str(model_dir)]
    assert len(env) == 1, 'log off unknown env'
    env = env[0]
    if env == 'dst':
        env = gym.make('DeepSeaTreasure-v0')
        env = TimeLimit(env, 200)
        max_treasure = np.amax(list(env.unwrapped._treasures().values()))
        max_return = np.array([max_treasure, -1.])

    elif env == 'minecart':
        env = gym.make('MinecartDeterministic-v0')
        env = TimeLimit(env, 1000)
        max_return = np.array([1.5, 1.5, -0.])

    elif env == 'sumo':
        q_range = 10
        env = gym.make('CrossroadSumo-v0')
        env = TimeLimit(env, max_episode_steps=100)
        env = FrameObservationEnv(env)
        env = CHWEnv(env)
        env = GrayscaleEnv(env)
        env = HistoryEnv(env, size=4)
        env = ScaleRewardEnv(env, min_=np.array([1.2, -0.9]), scale=90/q_range)
        max_return = np.array([1.5, 1.5])*q_range
    objectives = range(max_return.shape[-1])
        
    # ===================================
    # LOAD SAVED MODEL
    # ===================================
    log = model_dir / 'log.h5'
    log = h5py.File(log)
    checkpoints = [str(p) for p in model_dir.glob('model_100.pt')]
    checkpoints = sorted(checkpoints)
    model = torch.load(checkpoints[-1])
        
    # ===================================
    # LOAD SAVED ESTIMATE OF PARETO FRONT
    # ===================================
    with log:
        pareto_front = log['train/leaves/r/ndarray'][-1]
        pareto_front_h = log['train/leaves/h/ndarray'][-1]
        _, pareto_front_i = non_dominated(pareto_front[:,objectives], return_indexes=True)
        pareto_front = pareto_front[pareto_front_i]
        pareto_front_h = pareto_front_h[pareto_front_i]

        pf = np.argsort(pareto_front, axis=0)
        pareto_front = pareto_front[pf[:,0]]
        pareto_front_h = pareto_front_h[pf[:,0]]
        

    # ===================================
    # EVAL LOOP
    # ===================================
    inp = -1
    if not args.interactive:
        (model_dir / 'policies-executions').mkdir(exist_ok=True)
        print('='*38)
        print('not interactive, this may take a while')
        print('='*38)
        all_returns = []
    while True:
        if args.interactive:
            print('solutions: ')
            for i, p in enumerate(pareto_front):
                print(f'{i} : {p[objectives]}')
            inp = input('-> ')
            inp = int(inp)
        else:
            inp = inp+1
            if inp >= len(pareto_front):
                break
        desired_return = pareto_front[inp]
        desired_horizon = pareto_front_h[inp]

        # assume deterministic env, one run is enough
        transitions = run_episode(env, model, desired_return, desired_horizon, max_return)
        # compute return
        gamma = 1
        for i in reversed(range(len(transitions)-1)):
            transitions[i].reward += gamma * transitions[i+1].reward
        return_ = transitions[0].reward.flatten()
        print(f'ran model with desired-return: {desired_return.flatten()}, got {return_}')
        if not args.interactive:
            with open(model_dir / 'policies-executions' / f'policy_{inp}.txt', 'w') as f:
                f.write(', '.join(str(r) for r in return_))
