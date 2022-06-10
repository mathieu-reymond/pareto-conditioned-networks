import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from mones.metrics import non_dominated, compute_hypervolume
from logger import Logger
import torch.multiprocessing as mp


class Multiprocessor(object):

    def __init__(self):
        self.processes = []
        self.queue = mp.Manager().Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        torch.set_num_threads(2)
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = mp.Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets



def n_parameters(model):
    return np.sum([torch.prod(torch.tensor(p.shape)) for p in model.parameters()])


def set_parameters(model, z):
    assert len(z) == n_parameters(model), 'not providing correct amount of parameters'
    s = 0
    for p in model.parameters():
        n_p = torch.prod(torch.tensor(p.shape))
        p.data = z[s:s+n_p].view_as(p)
        s += n_p


def run_episode(env, model):

    e_r = 0; done = False
    o = env.reset()
    actions = []
    while not done:
        with torch.no_grad():
            actor_out = model(torch.from_numpy(o).float()[None,:])
            action = actor_out
            action = torch.argmax(actor_out, 1)

            action = action.numpy().flatten()[0]
            actions.append(action)

        n_o, r, done, _ = env.step(action)
        e_r += r
        o = n_o

    return torch.from_numpy(e_r).float()


def run_n_episodes(env, model, n):
    returns = [run_episode(env, model) for _ in range(n)]
    return torch.mean(torch.stack(returns, dim=0), dim=0)


def indicator_hypervolume(points, ref, nd_penalty=0.):
    # compute hypervolume of dataset
    nd = non_dominated(points)
    hv = compute_hypervolume(nd, ref)
    # hypervolume without a point from dataset
    hv_p = np.zeros(len(points))
    # penalization if point is dominated
    is_nd = np.zeros(len(points), dtype=bool)
    is_unique = np.zeros(len(points), dtype=bool)
    is_unique[np.unique(points, axis=0, return_index=True)[1]] = True
    for i in range(len(points)):
        is_nd[i] = np.any(np.all(nd == points[i], axis=1))
        if is_nd[i]:
            # if there is only one non-dominated point, and this point is non-dominated,
            # then it amounts for the full hypervolume
            if len(nd) == 1:
                hv_p[i] = 0.
            else:
                # remove point from nondominated points, compute hv
                rows = np.all(nd == points[i], axis=1)
                hv_p[i] = compute_hypervolume(nd[np.logical_not(rows)], ref)
        # if point is dominated, no impact on hypervolume
        else:
            hv_p[i] = hv

    indicator = hv - hv_p - nd_penalty*np.logical_not(is_nd)
    return indicator


class MONES(object):

    def __init__(self,
                 make_env,
                 policy,
                 n_population=1,
                 n_runs=1,
                 indicator='hypervolume',
                 ref_point=None,
                 lr=1e-1,
                 n_processes=1,
                 logdir='runs'):
        self.make_env = make_env
        self.policy = policy
        self.lr = lr

        self.n_population = n_population
        self.n_runs = n_runs
        self.ref_point = ref_point
        env = make_env()
        self.n_objectives = 1 if not hasattr(env, 'reward_space') else len(env.reward_space.low)

        self.logdir = logdir
        self.logger = Logger(self.logdir)

        self.n_processes = n_processes

        if indicator == 'hypervolume':
            assert ref_point is not None, 'reference point is needed for hypervolume indicator'
            self.indicator = lambda points, ref=ref_point: indicator_hypervolume(points, ref, nd_penalty=0.1)
        else:
            raise ValueError('unknown indicator, choose between hypervolume and non_dominated')
    
    def start(self):
        # make distribution
        n_params = n_parameters(self.policy)
        mu, sigma = torch.rand(n_params, requires_grad=True)*0.01, torch.rand(n_params, requires_grad=True)
        mu, sigma = nn.Parameter(mu), nn.Parameter(sigma)
        self.dist = torch.distributions.Normal(mu, sigma)

        # optimizer to change distribution parameters
        self.opt = torch.optim.Adam([{'params': mu}, {'params': sigma}], lr=self.lr)

    def step(self):
        # using current theta, sample policies from Normal(theta)
        population, z = self.sample_population()
        # # run episode for these policies
        # mproc = Multiprocessor()
        # # make a process per agent, that will learn concurrently
        # segments = np.linspace(0, self.n_population, self.n_processes+1).astype(int)
        # for i in range(self.n_processes):
        #     mproc.run(self.evaluate_population, self.make_env(), population[segments[i]:segments[i+1]])
        # returns = mproc.wait()
        # returns = torch.cat(returns, dim=0)

        returns = self.evaluate_population(self.make_env(), population)
        returns = returns.detach().numpy()

        indicator_metric = self.indicator(returns)
        metric = torch.tensor(indicator_metric)[:,None]

        # use fitness ranking TODO doesn't help
        # returns = centered_ranking(returns)
        # standardize the rewards to have a gaussian distribution
        metric = (metric - torch.mean(metric, dim=0)) / torch.std(metric, dim=0)

        # compute loss
        log_prob = self.dist.log_prob(z).sum(1, keepdim=True)

        mu, sigma = self.dist.mean, self.dist.scale

        # directly compute inverse Fisher Information Matrix (FIM)
        # only works because we use gaussian (and no correlation between variables) 
        fim_mu_inv = torch.diag(sigma.detach()**2)
        fim_sigma_inv = torch.diag(2/sigma.detach()**2)

        loss = -log_prob*metric

        # update distribution parameters
        self.opt.zero_grad()
        loss.mean().backward()
        # now that we have grads, multiply them with FIM_INV
        nat_grad_mu = fim_mu_inv@mu.grad
        nat_grad_sigma = fim_sigma_inv@sigma.grad
        mu.grad =  nat_grad_mu
        sigma.grad = nat_grad_sigma

        self.opt.step()

        return {'returns': returns, 'metric': np.mean(indicator_metric)}

    def train(self, iterations):
        self.start()

        for i in range(iterations):
            info = self.step()
            returns = info['returns']
            # logging
            self.logger.put('train/metric', info['metric'], i, 'scalar')
            self.logger.put('train/returns', returns, i, f'{returns.shape[-1]}d')
            if self.ref_point is not None:
                try:
                    hv = compute_hypervolume(returns, self.ref_point)
                except ValueError:
                    hv = 0.
                self.logger.put('train/hypervolume', hv, i, 'scalar')

            print(f'Iteration {i} \t Metric {info["metric"]} \t')
            print(non_dominated(returns))
            if (i+1)%20 == 0:
                torch.save(self.dist, self.logger.logdir / f'checkpoint_{i}.pt')

        print('='*20)
        print('DONE TRAINING, LAST POPULATION ND RETURNS')
        print(non_dominated(returns))
            

    def sample_population(self):
        population = []; z = []
        for _ in range(self.n_population):
            z_i = self.dist.sample()
            m_i = copy.deepcopy(self.policy)
            set_parameters(m_i, z_i)
            population.append(m_i)
            z.append(z_i)
        return population, torch.stack(z)

    def evaluate_population(self, env, population):
        returns = torch.zeros(len(population), self.n_objectives)
        for i in range(len(population)):
            p_return = torch.zeros(self.n_runs, self.n_objectives)
            for r in range(self.n_runs):
                p_return[r] = run_episode(env, population[i])
            returns[i] = torch.mean(p_return, dim=0)
        return returns

