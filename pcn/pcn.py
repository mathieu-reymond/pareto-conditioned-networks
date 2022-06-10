import heapq
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygmo import hypervolume
from logger import Logger


def crowding_distance(points):
    # first normalize accross dimensions
    points = (points-points.min(axis=0))/(points.ptp(axis=0)+1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding


@dataclass
class Transition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool

device = 'cpu'

def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    return is_efficient

def compute_hypervolume(q_set, ref):
    nA = len(q_set)
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

def nlargest(n, experience_replay, threshold=.2):
    returns = np.array([e[2][0].reward for e in experience_replay])
    # crowding distance of each point, check ones that are too close together
    distances = crowding_distance(returns)
    sma = np.argwhere(distances <= threshold).flatten()

    nd_i = get_non_dominated(returns)
    nd = returns[nd_i]
    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of nd to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(nd), 1))
    # distance to closest nd point
    l2 = np.min(np.linalg.norm(returns_exp-nd, axis=-1), axis=-1)*-1
    # all points that are too close together (crowding distance < threshold) get a penalty
    nd_i = np.nonzero(nd_i)[0]
    _, unique_i = np.unique(nd, axis=0, return_index=True)
    unique_i = nd_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] *= 2

    sorted_i = np.argsort(l2)
    largest = [experience_replay[i] for i in sorted_i[-n:]]
    # before returning largest elements, update all distances in heap
    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)
    return largest

def add_episode(transitions, experience_replay, gamma=1., max_size=100, step=0):
    # compute return
    for i in reversed(range(len(transitions)-1)):
        transitions[i].reward += gamma * transitions[i+1].reward
    # pop smallest episode of heap if full, add new episode
    # heap is sorted by negative distance, (updated in nlargest)
    # put positive number to ensure that new item stays in the heap
    if len(experience_replay) == max_size:
        heapq.heappushpop(experience_replay, (1, step, transitions))
    else:
        heapq.heappush(experience_replay, (1, step, transitions))

def choose_action(model, obs, desired_return, desired_horizon):
    log_probs = model(torch.tensor([obs]).to(device),
                      torch.tensor([desired_return]).to(device),
                      torch.tensor([desired_horizon]).unsqueeze(1).to(device))
    log_probs = log_probs.detach().cpu().numpy()[0]
    action = np.random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
    return action

def run_episode(env, model, desired_return, desired_horizon, max_return):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon)
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
        desired_return = np.clip(desired_return-reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))
    return transitions

def choose_commands(experience_replay, n_episodes):
    # get best episodes, according to their crowding distance
    episodes = nlargest(n_episodes, experience_replay)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    # keep only non-dominated returns
    nd_i = get_non_dominated(np.array(returns))
    returns = np.array(returns)[nd_i]
    horizons = np.array(horizons)[nd_i]
    # pick random return from random best episode
    r_i = np.random.randint(0, len(returns))
    desired_horizon = np.float32(horizons[r_i]-2)
    # mean and std per objective
    m, s = np.mean(returns, axis=0), np.std(returns, axis=0)
    # desired return is sampled from [M, M+S], to try to do better than mean return
    desired_return = returns[r_i].copy()
    # random objective
    r_i = np.random.randint(0, len(desired_return))
    desired_return[r_i] += np.random.uniform(high=s[r_i])
    desired_return = np.float32(desired_return)
    return desired_return, desired_horizon

def update_model(model, opt, experience_replay, batch_size):
    batch = []
    # randomly choose episodes from experience buffer
    s_i = np.random.choice(np.arange(len(experience_replay)), size=batch_size, replace=True)
    for i in s_i:
        # episode is tuple (return, transitions)
        ep = experience_replay[i][2]
        # choose random timestep from episode, 
        # use it's return and leftover timesteps as desired return and horizon
        t = np.random.randint(0, len(ep))
        # reward contains return until end of episode
        s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep)-t)
        batch.append((s_t, a_t, r_t, h_t))

    obs, actions, desired_return, desired_horizon = zip(*batch)
    log_prob = model(torch.tensor(obs).to(device),
                     torch.tensor(desired_return).to(device),
                     torch.tensor(desired_horizon).unsqueeze(1).to(device))

    opt.zero_grad()
    # one-hot of action for CE loss
    actions = F.one_hot(torch.tensor(actions).long().to(device), len(log_prob[0]))
    # cross-entropy loss
    l = torch.sum(-actions*log_prob, -1)
    l = l.mean()
    l.backward()
    opt.step()

    return l, log_prob


def eval(env, model, experience_replay, max_return, gamma=1., n=10):
    episodes = nlargest(n, experience_replay)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    returns = np.float32(returns); horizons = np.float32(horizons)
    e_returns = []
    for i in range(n):
        transitions = run_episode(env, model, returns[i], np.float32(horizons[i]-2), max_return)
        # compute return
        for i in reversed(range(len(transitions)-1)):
            transitions[i].reward += gamma * transitions[i+1].reward
        e_returns.append(transitions[0].reward)

    e_returns = np.array(e_returns)
    distances = np.linalg.norm(np.array(returns)-e_returns, axis=-1)
    return e_returns, np.array(returns), distances


def train(env, 
          model,
          learning_rate=1e-2,
          batch_size=1024, 
          total_steps=1e7,
          n_model_updates=100,
          n_step_episodes=10,
          n_er_episodes=500,
          gamma=1.,
          max_return=250.,
          max_size=500,
          ref_point=np.array([0, 0]),
          logdir='runs/'):
    step = 0
    total_episodes = n_er_episodes
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = Logger(logdir=logdir)
    n_checkpoints = 0
    # fill buffer with random episodes
    experience_replay = []
    for _ in range(n_er_episodes):
        transitions = []
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, env.nA)
            n_obs, reward, done, _ = env.step(action)
            transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
            step += 1
        # add episode in-place
        add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
    while step < total_steps:
        loss = []
        entropy = []
        for _ in range(n_model_updates):
            l, lp = update_model(model, opt, experience_replay, batch_size=batch_size)
            loss.append(l.detach().cpu().numpy())
            lp = lp.detach().cpu().numpy()
            ent = np.sum(-np.exp(lp)*lp)
            entropy.append(ent)

        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes)

         # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
        leaves_r = np.array([e[2][0].reward for e in experience_replay[len(experience_replay)//2:]])
        leaves_h = np.array([len(e[2]) for e in experience_replay[len(experience_replay)//2:]])
        try:
            if len(experience_replay) == max_size:
                logger.put('train/leaves/r', leaves_r, step, f'{leaves_r.shape[-1]}d')
                logger.put('train/leaves/h', leaves_h, step, f'{leaves_h.shape[-1]}d')
            hv = hypervolume(leaves_r*-1)
            hv_est = hv.compute(ref_point*-1)
            logger.put('train/hypervolume', hv_est, step, 'scalar')
        except ValueError:
            pass

        returns = []
        horizons = []
        for _ in range(n_step_episodes):
            transitions = run_episode(env, model, desired_return, desired_horizon, max_return)
            step += len(transitions)
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))
        
        total_episodes += n_step_episodes
        logger.put('train/episode', total_episodes, step, 'scalar')
        logger.put('train/loss', np.mean(loss), step, 'scalar')
        logger.put('train/entropy', np.mean(entropy), step, 'scalar')
        logger.put('train/horizon/desired', desired_horizon, step, 'scalar')
        logger.put('train/horizon/distance', np.linalg.norm(np.mean(horizons)-desired_horizon), step, 'scalar')
        for o in range(len(desired_return)):
            logger.put(f'train/return/{o}/value', desired_horizon, step, 'scalar')
            logger.put(f'train/return/{o}/desired', np.mean(np.array(returns)[:, o]), step, 'scalar')
            logger.put(f'train/return/{o}/distance', np.linalg.norm(np.mean(np.array(returns)[:, o])-desired_return[o]), step, 'scalar')
        print(f'step {step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}')
        if step >= (n_checkpoints+1)*total_steps/100:
            torch.save(model, f'{logger.logdir}/model_{n_checkpoints+1}.pt')
            n_checkpoints += 1

            e_r, e_dr, e_d = eval(env, model, experience_replay, max_return, gamma=gamma)
            s = 'desired return vs evaluated return\n'+33*'='+'\n'
            for i in range(len(e_r)):
                s += f'{e_dr[i]}  \t  {e_r[i]}  \n'
            logger.put('eval/return/desired', e_dr, step, f'{len(desired_return)}d')
            logger.put('eval/return/value', e_r, step, f'{len(desired_return)}d')
            for o in range(len(desired_return)):
                logger.put(f'eval/return/{o}/distance', e_d[o], step, 'scalar')
