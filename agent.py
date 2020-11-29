############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

from collections import deque
import numpy as np
import torch
import time
from matplotlib import pyplot as plt

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=32)
        self.layer_2 = torch.nn.Linear(in_features=32, out_features=32)
        self.output_layer = torch.nn.Linear(in_features=32, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


class ReplayBuffer:

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size, p=None):
        minibatch = []
        if p:
            indices = np.random.choice(len(self.buffer), size=batch_size, p=p)
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size)
        for idx in indices:
            minibatch.append(self.buffer[idx])
        return minibatch

    def curr_size(self):
        return len(self.buffer)


class Agent:

    # Function to initialise the agent
    def __init__(self):
        self.epsilon = 0.5
        self.old_epsilon = None
        self.decay = 0.95
        self.gamma = 0.9
        self.lr = 0.001
        self.batch_size = 64
        self.update_target_freq = 50
        # Set the episode length
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        self.replay_buffer = ReplayBuffer(5000)
        self.weights = []
        self.probs = []
        self.alpha = 0.5

        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        elif discrete_action == 2:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 3:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        return continuous_action

    def _continous_action_to_discrete(self, continuous_action):
        e = 0.001
        if continuous_action[0] > e:
            discrete_action = 0
        elif continuous_action[0] < 0:
            discrete_action = 1
        if continuous_action[1] > e:
            discrete_action = 2
        elif continuous_action[1] < 0:
            discrete_action = 3
        return discrete_action

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        action_space = [0, 1, 2, 3]
        k = len(action_space)
        action = None
        epsilon_greedy = False

        if self.state is not None:
            greedy_action = self.get_greedy_action(self.state, index=True)
            epsilon_greedy = True

        if epsilon_greedy:
            probabilities = [self.epsilon / k] * k
            probabilities[greedy_action] = 1 - self.epsilon + (self.epsilon / k)
            rand = np.random.rand()
            action, curr_p = None, 0
            for i, p in enumerate(probabilities):
                curr_p += p
                if rand < curr_p:
                    action = i
                    break
        else:
            # Here, the action is random, but you can change this
            action = np.random.choice(action_space)

        action = self._discrete_action_to_continuous(action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    def _calculate_loss(self, states, actions, rewards, next_states):
        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_prediction = self.q_network.forward(states)
        q_prediction = q_prediction.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        q_label = self.target_network.forward(next_states).detach()
        q_label, idx = q_label.max(1)
        q_label = self.q_network.forward(next_states).gather(1, idx.unsqueeze(-1)).squeeze(-1)
        # q_label = q_label.gather(1, max_actions.unsqueeze(-1)).squeeze(-1).detach()
        return torch.nn.MSELoss()(q_prediction, rewards + (self.gamma * q_label))

    def _train_q_network(self, transitions):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(*transitions)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def _update_target_network(self):
        q_network_state = self.q_network.state_dict()
        self.target_network.load_state_dict(q_network_state)

    def _calculate_weight(self, state, action, reward, next_state):
        e = 0.01
        action = torch.tensor(action).unsqueeze(0).unsqueeze(0)
        state = torch.tensor(state).unsqueeze(0)
        next_state = torch.tensor(next_state).unsqueeze(0)

        q_prediction = self.q_network.forward(state).detach()
        q_prediction = q_prediction.gather(1, action)

        q_label = self.target_network.forward(next_state).detach()
        q_label = q_label.max(1)[0]

        delta = np.abs(reward + q_label - q_prediction)[0][0].item()
        return delta + e

    def _update_weights_probabilities(self):
        for i, transition in enumerate(self.replay_buffer.buffer):
            self.weights[i] = self._calculate_weight(*transition)

        weight_sum = sum((w**self.alpha for w in self.weights))
        for i, w in enumerate(self.weights):
            self.probs[i] = w**self.alpha / weight_sum

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        penalty = 1
        self.epsilon *= self.decay
        if np.array_equal(self.state, next_state):
            pentalty = 1.5
            self.epsilon /= self.decay
        reward = 1 - (distance_to_goal * penalty)

        episode_steps = self.num_steps_taken % self.episode_length
        if self.has_finished_episode:
            self.train = True
        elif self.epsilon == 0 and self.has_finished_episode:
            self.epsilon = self.old_epsilon
        elif self.epsilon == 0 and distance_to_goal < 0.03 and episode_steps < 100:
            self.epsilon = self.old_epsilon
            self.train = False
        elif distance_to_goal < 0.05 and self.epsilon <= 0.1 and episode_steps < 100:
            self.old_epsilon = self.epsilon
            self.epsilon = 0

        # Create a transition
        discrete_action = self._continous_action_to_discrete(self.action)
        transition = (self.state, discrete_action, reward, next_state)
        self.replay_buffer.add(transition)
        # Now you can do something with this transition ...
        if self.weights:
            max_w = max(self.weights)
            self.probs.append(0)
        else:
            max_w = self._calculate_weight(*transition)
            self.probs.append(1)
        self.weights.append(max_w)

        if self.replay_buffer.curr_size() >= self.batch_size and self.train:
            batch = self.replay_buffer.sample(self.batch_size, p=self.probs)
            states, actions, rewards, next_states = [], [], [], []

            for state, action, reward, next_st in batch:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_st)

            states, actions = np.array(states), np.array(actions)
            rewards, next_state = np.array(rewards), np.array(next_states)
            loss = self._train_q_network((states, actions, rewards, next_states))

        self._update_weights_probabilities()
        if self.num_steps_taken % self.update_target_freq == 0:
            self._update_target_network()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state, index=False):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        curr_state = torch.from_numpy(np.array(state)).unsqueeze(0)
        action = np.argmax(self.q_network.forward(curr_state).detach())
        if index:
            return action
        else:
            return self._discrete_action_to_continuous(action)
