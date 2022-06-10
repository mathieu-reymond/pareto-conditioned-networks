import torch


class Categorical(object):

    def __call__(self, log_probs, log=None):
        probs = torch.exp(log_probs)
        actions = torch.multinomial(probs, 1)
        return actions

    def log_prob(self, samples, log_probs):
        return log_probs.gather(1, samples)