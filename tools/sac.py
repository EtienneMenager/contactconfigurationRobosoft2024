###https://github.com/jparkerholder/SAC-PyTorch
###https://github.com/kengz/SLM-Lab/blob/b8696faeb4cd8bf105d2c4abaa8936bd953fd0a7/slm_lab/agent/algorithm/sac.py

import itertools
import math
import os
import random
from collections import deque, namedtuple

import numpy as np
import copy

import torch
import torch.nn as nn
from torch import distributions
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, Categorical
import pickle

device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))

class MeanStdevFilter():
    def __init__(self, shape, clip=3.0):
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = np.zeros(shape)
        self.stdev = np.ones(shape) * self.eps

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)

        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))

    def __call__(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean

class ReplayPool:
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))

    def push(self, transition):
        """ Saves a transition """
        self._memory.append(transition)

    def sample(self, batch_size: int):
        transitions = random.sample(self._memory, batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int):
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self):
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def save_pool(self, filename):
        filehandler = open(filename, 'wb')
        pickle.dump(self._memory, filehandler)

    def load_pool(self, filename):
        filehandler = open(filename, 'rb')
        self._memory = pickle.load(filehandler)

# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )

    def forward(self, x):
        return self.network(x)

class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


def create_actor_distribution(actor_output):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    action_distribution = GumbelSoftmax(torch.tensor(1.0), logits=actor_output)
    return action_distribution

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, continuous = True):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.continuous = continuous

        if self.continuous:
            self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)
        else:
            self.network = MLPNetwork(state_dim, action_dim, hidden_size)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, get_logprob=False, reparam=False):
        if self.continuous:
            mu_logstd = self.network(x)
            mu, logstd = mu_logstd.chunk(2, dim=1)

            #classical logstd of the distribution, put in [-20, 2]
            logstd = torch.clamp(logstd, -20, 2)
            std = logstd.exp()
            dist = Normal(mu, std)
            transforms = [TanhTransform(cache_size=1)]
            dist = TransformedDistribution(dist, transforms)
            action = dist.rsample()

            if get_logprob:
                logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
            else:
                logprob = None

            mean = torch.tanh(mu)
            #action = torch.tanh(action)

            return action, logprob, mean
        else:
            pdparams = self.network(x)
            action_probabilities = self.softmax(pdparams)
            max_probability_action = torch.argmax(action_probabilities, dim=-1)

            action_pd = create_actor_distribution(pdparams)
            actions = action_pd.rsample() if reparam else action_pd.sample()
            log_prob = action_pd.log_prob(actions)
            return actions, log_prob, action_probabilities, max_probability_action


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, continuous = True):
        super(DoubleQFunc, self).__init__()
        self.action_dim = action_dim
        self.continuous = continuous
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def guard_q_actions(self, actions):
        '''Guard to convert actions to one-hot for input to Q-network'''
        actions = F.one_hot(actions.long(), self.action_dim).float()
        return actions

    def forward(self, state, action):
        if not self.continuous and len(action.size())<=1:
            action = self.guard_q_actions(action)
            if len(action.size())==1:
                action = action.view(1, -1)
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class SAC_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, seed=0, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256, update_interval=1, continuous = True, bufferSize = int(1e6), use_alpha = None):
        super(SAC_Agent, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.continuous = continuous
        if self.continuous:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = -np.log((1.0 / action_dim)) * 0.98
        self.batchsize = batchsize
        self.update_interval = update_interval

        #torch.manual_seed(seed)

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size, continuous=self.continuous).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size, continuous = self.continuous).to(device)

        # aka temperature
        self.use_alpha = use_alpha
        if self.use_alpha is None:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = use_alpha

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = ReplayPool(capacity= bufferSize)

    def save_pool(self,file = './Results', name = 'sac_pool'):
        filename = file + '/' + name + '.txt'
        self.replay_pool.save_pool(filename)
        print(">>    Pool saved at {}".format(filename))

    def load_pool(self,file = './Results', name = 'sac_pool'):
        filename = file + '/' + name + '.txt'
        self.replay_pool.load_pool(filename)
        print(">>    Pool loaded at {}".format(filename))

    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter is not None:
            state = state_filter(state)

        if self.continuous:
            with torch.no_grad():
                action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(device))
            if deterministic:
                return np.atleast_1d(mean.squeeze().cpu().numpy())
            return np.atleast_1d(action.squeeze().cpu().numpy())
        else:
            with torch.no_grad():
                action, _, _, max_probability_action = self.policy(torch.Tensor(state).view(1,-1).to(device))
            if deterministic:
                return max_probability_action.squeeze().cpu().numpy()
            return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch, reparam=False):
        with torch.no_grad():
            if self.continuous:
                nextaction_batch, next_log_probs, _ = self.policy(nextstate_batch, get_logprob=True)
                q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            else:
                nextaction_batch, next_log_probs, _, _ = self.policy(nextstate_batch, get_logprob=True)
                q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
                next_log_probs = next_log_probs.view(-1, 1)
            next_target_q_preds = torch.min(q_t1, q_t2)
            q_inter = next_target_q_preds - self.alpha * next_log_probs
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_inter)

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy_and_temp(self, state_batch):
        if self.continuous:
            action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True, reparam = True)
        else:
            action_batch, logprobs_batch, _, _ = self.policy(state_batch, get_logprob=True, reparam = True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)

        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        logpi = logprobs_batch
        if self.use_alpha is None:
            temp_loss = -self.log_alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        else:
            temp_loss = None

        return policy_loss, temp_loss, logpi

    def optimize(self, n_updates, state_filter=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)

            if type(samples.state[0])==np.array:
                if state_filter is not None:
                    state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                    nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
                else:
                    state_batch = torch.FloatTensor(samples.state).to(device)
                    nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            else:
                state_batch = torch.cat(samples.state, dim = 0).to(device)
                nextstate_batch = torch.cat(samples.nextstate, dim = 0).to(device)

            action_batch = torch.FloatTensor(np.array(samples.action)).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.done).to(device).unsqueeze(1)

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch, nextstate_batch, done_batch)

            q_loss_step = q1_loss_step + q2_loss_step


            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step, _ = self.update_policy_and_temp(state_batch)

            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()

            if self.use_alpha is None:
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()
                self.alpha = self.log_alpha.exp()
                a_loss += a_loss_step.detach().item()/n_updates

            for p in self.q_funcs.parameters():
                p.requires_grad = True

            q1_loss += q1_loss_step.detach().item()/n_updates
            q2_loss += q2_loss_step.detach().item()/n_updates
            pi_loss += pi_loss_step.detach().item()/n_updates

            if i % self.update_interval == 0:
                self.update_target()

        return q1_loss, q2_loss, pi_loss, a_loss

    def save_model(self, file = './Results', name = 'sac_latest'):
        path_model = file + '/' + name +'.pth'
        os.makedirs(file , exist_ok=True)
        torch.save(self.state_dict(), path_model)

        print(">>    Model saved at {}".format(path_model))

    def load_model(self, file = './Results', name = 'sac_latest'):
        path_model = file + '/' + name +'.pth'
        self.load_state_dict(torch.load(path_model), strict=False)
        print(">>    Model loaded at {}".format(path_model))

def func_reward(reward, observation, tol = 0.02, target = None):
    if target is None:
        reward = reward
    else:
        if abs(observation[2]-target)<tol:
            reward = 1
        else:
            reward = 0.1
    return reward

def evaluate_agent(env, agent, state_filter, n_starts=1, render = False, CONTINUOUS = False, target = None):
    reward_sum, len_sum = 0, 0
    liste_reward = []
    for _ in range(n_starts):
        done = False
        state = env.reset()
        state =  torch.tensor(state).view(1,-1).type(torch.float)
        r = 0
        l = 0
        r_pos = 0
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            if not CONTINUOUS:
                reward = func_reward(reward, nextstate, target = target)
            if render:
                env.render()
            if reward > 0:
                r_pos +=1
            reward_sum += reward
            r += reward
            state = nextstate
            state =  torch.tensor(state).view(1,-1).type(torch.float)
            l+=1
        len_sum+=l
        liste_reward.append(r)
    return reward_sum / n_starts, liste_reward, len_sum/n_starts

indexedTransition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done', 'idNode','toNode'))

class indexedReplayPool:
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))

    def push(self, transition):
        """ Saves a transition """
        self._memory.append(transition)

    def sample(self, batch_size: int):
        transitions = random.sample(self._memory, batch_size)
        return indexedTransition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int):
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return indexedTransition(*zip(*transitions))

    def get_all(self):
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def save_pool(self, filename):
        filehandler = open(filename, 'wb')
        pickle.dump(self._memory, filehandler)

    def load_pool(self, filename):
        filehandler = open(filename, 'rb')
        self._memory = pickle.load(filehandler)

class InsideSac(SAC_Agent):
    def __init__(self,state_dim, action_dim, seed=0, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256, update_interval=1, continuous = True, bufferSize = int(1e6), use_alpha = None):
        super(InsideSac, self).__init__(state_dim, action_dim, seed=seed, lr=lr, gamma=gamma, tau=tau, batchsize=batchsize, hidden_size=hidden_size, update_interval=update_interval, continuous = continuous, bufferSize = bufferSize, use_alpha = use_alpha)
        self.replay_pool = indexedReplayPool(capacity= bufferSize)

    def addTransition(self, state, action, reward, nextstate, done, idNode, toNode):
        self.replay_pool.push(indexedTransition(state, action, reward, nextstate, done, idNode, toNode))

    def computeQval(self, next_state, id_node, id_next_node, grl):
        if id_next_node ==  id_node:
            with torch.no_grad():
                node = grl.index_to_nodes[id_node]
                if self.continuous:
                    a, log_prob, _ = node.inside_agent.policy(next_state, get_logprob=True)
                else:
                    a, log_prob, _, _ = node.inside_agent.policy(next_state, get_logprob=True)
                q_t1, q_t2 = node.inside_agent.target_q_funcs(next_state, a)
            q_val = torch.min(q_t1, q_t2)
        else:
            with torch.no_grad():
                node = grl.index_to_nodes[id_next_node]
                if self.continuous:
                    a, log_prob, _ = node.inside_agent.policy(next_state, get_logprob=True)
                else:
                    a, log_prob, _, _ = node.inside_agent.policy(next_state, get_logprob=True)
                q_t1, q_t2 = node.inside_agent.target_q_funcs(next_state, a)
            q_val = torch.min(q_t1, q_t2)
        return [q_val.item()], [log_prob.item()]

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch, idNode_batch, toNode_batch, grl):
        q_val, log_prob = [], []
        for i, id in enumerate(toNode_batch):
            q, l = self.computeQval(nextstate_batch[i].view(1, -1), idNode_batch[i], id, grl)
            q_val.append(q)
            log_prob.append(l)

        next_target_q_preds = torch.tensor(q_val)
        next_log_probs = torch.tensor(log_prob)
        q_inter = next_target_q_preds - self.alpha * next_log_probs
        value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_inter)

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def optimize(self, n_updates, grl, state_filter=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)

            if type(samples.state[0])==np.array:
                if state_filter is not None:
                    state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                    nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
                else:
                    state_batch = torch.FloatTensor(samples.state).to(device)
                    nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            else:
                state_batch = torch.cat(samples.state, dim = 0).to(device)
                nextstate_batch = torch.cat(samples.nextstate, dim = 0).to(device)

            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.done).to(device).unsqueeze(1)
            toNode_batch = samples.toNode
            idNode_batch = samples.idNode

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch, nextstate_batch, done_batch, idNode_batch, toNode_batch, grl)
            q_loss_step = q1_loss_step + q2_loss_step

            self.q_optimizer.zero_grad()
            q_loss_step.backward(retain_graph=True)
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step, _ = self.update_policy_and_temp(state_batch)

            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()

            if self.use_alpha is None:
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()
                self.alpha = self.log_alpha.exp()
                a_loss += a_loss_step.detach().item()/n_updates

            for p in self.q_funcs.parameters():
                p.requires_grad = True

            q1_loss += q1_loss_step.detach().item()/n_updates
            q2_loss += q2_loss_step.detach().item()/n_updates
            pi_loss += pi_loss_step.detach().item()/n_updates

            if i % self.update_interval == 0:
                self.update_target()

        return q1_loss, q2_loss, pi_loss, a_loss


class EvaluatorPolicy(nn.Module):
    def __init__(self, d_input, d_q, hidden_size=512, type_attention_evaluator = "H"):
        super(EvaluatorPolicy, self).__init__()
        self.d_q = d_q
        self.Fq = MLPNetwork(d_input, 2*d_q, hidden_size=hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = np.power(d_q, 0.5)
        self.type_attention_evaluator = type_attention_evaluator
        if self.type_attention_evaluator == "Fk":
            self.Fk = nn.Linear(d_q, d_q)

    def createH(self, H, Id):
        self.H = H
        self.Id = Id

        if self.type_attention_evaluator == "Id" and self.Id.shape[1]!= self.d_q:
            print("[ERROR] >> You want to use Id encoding but the input of the encoding network doesn't match the size of Id. (size Id: {} - size input: {}). Please check the config to modify dq.".format(self.Id.shape[0], self.d_q))
            exit(1)
        elif self.type_attention_evaluator == "H" and self.H.shape[1]!= self.d_q:
            print(
                "[ERROR] >> You want to use H encoding but the input of the encoding network doesn't match the size of H. (size H: {} - size input: {}). Please check the config to modify dq.".format(
                    self.H.shape[0], self.d_q))
            exit(1)


    def compute_action(self, x):
        if len(x.size())==1:
            x = x.view(1, -1)

        if self.type_attention_evaluator == "Fk":
            K = self.Fk(self.H)
        elif self.type_attention_evaluator == "Id":
            K = self.Id
        else:
            K = self.H

        Q = self.Fq(x)
        Q_mean, Q_std = Q.chunk(2, dim=1)

        attn_mean = torch.matmul(Q_mean, K.transpose(1,0))
        attn_mean = attn_mean/self.temperature
        attn_std = torch.matmul(Q_std, K.transpose(1,0))

        if x.size()[0]==1:
            attn_std = attn_std.view(1, -1)
            attn_mean = attn_mean.view(1, -1)
        return attn_mean, attn_std

    def forward(self, x, get_logprob=False, reparam=False):
        mu, std = self.compute_action(x)

        #classical logstd of the distribution, put in [-20, 2]
        logstd = torch.clamp(std, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()


        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None

        mu = self.softmax(mu)
        action = self.softmax(action)

        return torch.matmul(action, self.Id), logprob, torch.matmul(mu,  self.Id)

    def get_action(self, x):
        action, _, mean = self.forward(x)
        return action, mean

class EvaluatorDoubleQFunc(nn.Module):
    def __init__(self, state_dim, d_v, hidden_size=256):
        super(EvaluatorDoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + d_v, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + d_v, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class EvaluatorSac(SAC_Agent):
    def __init__(self, state_dim, d_q, d_v, batchsize = 256,  hidden_size_policy=512, hidden_size_qfun=512, bufferSize = int(1e6), lr = 1e-4, use_alpha = None, gamma=0.99, tau=5e-3, type_attention_evaluator = "H"):
        super(EvaluatorSac, self).__init__(state_dim, d_v, batchsize = batchsize, bufferSize = bufferSize, use_alpha = use_alpha, gamma=gamma, tau=tau, continuous=True)
        self.policy = EvaluatorPolicy(state_dim, d_q, hidden_size=hidden_size_policy, type_attention_evaluator = type_attention_evaluator).to(device)
        self.q_funcs = EvaluatorDoubleQFunc(state_dim, d_v, hidden_size=hidden_size_qfun)

        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = indexedReplayPool(capacity= bufferSize)

    def addTransition(self, state, action, reward, nextstate, done, idNode, toNode):
        self.replay_pool.push(indexedTransition(state, action, reward, nextstate, done, idNode, toNode))

    def computeQval(self, next_state, idNode, id_next_node, grl):
        if idNode == id_next_node:
            with torch.no_grad():
                node = grl.index_to_nodes[idNode]
                a, log_prob, _ = node.evaluator.policy(next_state, get_logprob=True)
                q_t1, q_t2 = node.evaluator.target_q_funcs(next_state, a)
        else:
            with torch.no_grad():
                node = grl.index_to_nodes[id_next_node]
                a, log_prob, _ = node.evaluator.policy(next_state, get_logprob=True)
                q_t1, q_t2 = node.evaluator.target_q_funcs(next_state, a)
        q_val = torch.min(q_t1, q_t2)
        return [q_val.item()], [log_prob.item()]


    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch, idNode_batch, toNode_batch, grl):
        q_val, log_prob = [], []
        for i, id in enumerate(toNode_batch):
            q, l = self.computeQval(nextstate_batch[i].view(1, -1), idNode_batch[i], id, grl)
            q_val.append(q)
            log_prob.append(l)
        next_target_q_preds = torch.tensor(q_val)
        next_log_probs = torch.tensor(log_prob)

        q_inter = next_target_q_preds - self.alpha * next_log_probs
        value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_inter)

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)

        return loss_1, loss_2


    def optimize(self, n_updates, grl, state_filter=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)

            if type(samples.state[0])==np.array:
                if state_filter is not None:
                    state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                    nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
                else:
                    state_batch = torch.FloatTensor(samples.state).to(device)
                    nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            else:
                state_batch = torch.cat(samples.state, dim = 0).to(device)
                nextstate_batch = torch.cat(samples.nextstate, dim = 0).to(device)

            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.done).to(device).unsqueeze(1)
            toNode_batch = samples.toNode
            idNode_batch = samples.idNode

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch, nextstate_batch, done_batch, idNode_batch, toNode_batch, grl)
            q_loss_step = q1_loss_step + q2_loss_step

            self.q_optimizer.zero_grad()
            q_loss_step.backward(retain_graph=True)
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step, _ = self.update_policy_and_temp(state_batch)

            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()

            if self.use_alpha is None:
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()
                self.alpha = self.log_alpha.exp()
                a_loss += a_loss_step.detach().item()/n_updates

            for p in self.q_funcs.parameters():
                p.requires_grad = True

            q1_loss += q1_loss_step.detach().item()/n_updates
            q2_loss += q2_loss_step.detach().item()/n_updates
            pi_loss += pi_loss_step.detach().item()/n_updates

            if i % self.update_interval == 0:
                self.update_target()

        return q1_loss, q2_loss, pi_loss, a_loss

    def createH(self, H, Id):
        self.policy.createH(H, Id)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            action, mean = self.policy.get_action(torch.Tensor(state).view(1,-1).to(device))
        if deterministic:
            return mean.squeeze().cpu().numpy()
        return np.atleast_1d(action.squeeze().cpu().numpy())



if __name__ == "__main__":
    import gym
    from gym.wrappers import RescaleAction
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ls", "--list_s", help="The num of the test we want to plot",
                         action='append', required = True)
    parser.add_argument("-ag", "--angle", help="Use specific angle",
                         action="store_true")
    args = parser.parse_args()
    num = args.list_s[0]


    TRAIN = True
    CONTINUOUS = False
    TARGET = None #0.06


    if CONTINUOUS:
        env_name = 'Pendulum-v0'
    else:
        env_name = "CartPole-v0"

    seed = 10*int(num)
    use_statefilter, save_model = False, False
    update_timestep = 5
    n_random_actions = 500
    n_collect_steps = 1000
    n_evals = 10

    if args.angle:
        num = num + "_angle"

    env = gym.make(env_name)
    env_test = gym.make(env_name)
    if CONTINUOUS:
        env = RescaleAction(env, -1, 1)
        env_test = RescaleAction(env_test, -1, 1)


    state_dim = env.observation_space.shape[0]

    if CONTINUOUS:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    use_alpha = None
    agent = SAC_Agent(state_dim, action_dim, use_alpha = use_alpha, lr=1e-4,  gamma=0.95, tau=5e-3, hidden_size=512, seed = seed, continuous = CONTINUOUS)

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    log_interval = 1000
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps
    reward_list, step = [], []
    if TRAIN:
        while samples_number < 26000:
            time_step = 0
            episode_reward = 0
            i_episode += 1
            log_episode += 1

            state = env.reset()

            if state_filter:
                state_filter.update(state)

            done = False
            state =  torch.tensor(state).view(1,-1).type(torch.float)

            while (not done):
                cumulative_log_timestep += 1
                cumulative_timestep += 1
                time_step += 1
                samples_number += 1

                if samples_number < n_random_actions:
                    action = env.action_space.sample()
                else:
                    action = agent.get_action(state, state_filter=state_filter)
                    if not CONTINUOUS:
                        action = action[0]
                nextstate, reward, done, _ = env.step(action)
                if not CONTINUOUS:
                    reward = func_reward(reward, nextstate, target = TARGET)
                # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
                real_done = False if time_step == max_steps else done
                nextstate = torch.tensor(nextstate).view(1,-1).type(torch.float)
                agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
                state = nextstate
                if state_filter:
                    state_filter.update(state)

                episode_reward += reward
                # update if it's time
                if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                    q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(update_timestep, state_filter=state_filter)
                    n_updates += 1

                # logging
                if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                    avg_length = np.mean(episode_steps)
                    running_reward = np.mean(episode_rewards)
                    eval_reward, _, _ = evaluate_agent(env, agent, state_filter, n_starts=n_evals, target = TARGET)
                    print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
                    episode_steps = []
                    episode_rewards = []

                    agent.save_model()

                if samples_number%250==0:
                    current_reward, current_list_reward, current_len = evaluate_agent(env_test, agent, state_filter, n_starts=10, target = TARGET)
                    print("Sample:", samples_number, "VALIDATION:", current_reward, " - ", current_len)
                    reward_list += current_list_reward
                    step += [samples_number for _ in range(len(current_list_reward))]
                    with open("./Results/CartPol/sac/rewards_"+num+".txt", 'w') as fp:
                        json.dump([reward_list, step], fp)

                # if samples_number%2000==0:
                #     current_reward, _, _ = evaluate_agent(env, agent, state_filter, n_starts=1, render = True, target = TARGET)
                #     print("SUPER VALIDATION:", current_reward)

            episode_steps.append(time_step)
            episode_rewards.append(episode_reward)

            with open("./Results/CartPol/sac/rewards_"+num+".txt", 'w') as fp:
                json.dump([reward_list, step], fp)


    agent.load_model()
    r, _, _ = evaluate_agent(env, agent, state_filter, n_starts=10, target = TARGET)
    print(">>  REWARD IS:", r)
