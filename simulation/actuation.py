from telnetlib import SE
from tkinter import N
from tkinter.messagebox import NO
import torch
import torch.nn as nn

from common.networks import BaseNetwork, MultiLayerPerceptron, GatedRecurrentUnit


class ActuatorNetwork(BaseNetwork):
    def __init__(self, in_dim=2, hidden_state_dim=8, dropout=0.):
        super(ActuatorNetwork, self).__init__()

        # Set up the recurrent block of the actuator network
        self._recurrent_block = GatedRecurrentUnit(
            in_dim=in_dim, hidden_state_dim=hidden_state_dim
        )

        # Set up the dense block of the actuator network
        self._feed_forward_block = MultiLayerPerceptron(
            in_dim=in_dim + hidden_state_dim,
            hidden_layers=[hidden_state_dim, hidden_state_dim],
            out_dim=1, activation=nn.LeakyReLU, dropout=dropout
        )

        self._hidden_state_dim = hidden_state_dim
        self._hidden_state = torch.zeros((0, self._hidden_state_dim))
        self._hidden_state = self._hidden_state.to(
            next(self.parameters()).device)

    def forward(self, x):
        self._hidden_state = self._recurrent_block(x, self._hidden_state)
        return self._feed_forward_block(torch.cat([x, self._hidden_state], axis=1))

    def reset(self, batch_size=0, randomize=False):
        if randomize:
            self._hidden_state = torch.randn(
                (batch_size, self._hidden_state_dim)) * 0.1
        else:
            self._hidden_state = torch.zeros(
                (batch_size, self._hidden_state_dim))

        self._hidden_state = self._hidden_state.to(
            next(self.parameters()).device)

    @property
    def gru(self):
        return self._recurrent_block

    @property
    def mlp(self):
        return self._feed_forward_block

    def _apply(self, fn):
        super(ActuatorNetwork, self)._apply(fn)

        try:
            self._hidden_state = fn(self._hidden_state)

            self._feed_forward_block = fn(self._feed_forward_block)
            self._recurrent_block = fn(self._recurrent_block)

        except AttributeError as e:
            print('Warning:', e)

        return self


class Actuation:
    def __init__(self, parameters_path, actuator_network=None, device=torch.device('cpu'),
                 input_scaling=None, output_scaling=100.0, limit=80., num_actuators=12):

        self._parameters_path = parameters_path
        self._device = device

        self._actuator_network = actuator_network if actuator_network is not None else ActuatorNetwork()
        self._actuator_network.load_state_dict(torch.load(parameters_path))

        self._actuator_network.to(device)

        self._input_scaling = input_scaling if input_scaling is not None \
            else [1.0, 0.1]

        self._output_scaling = output_scaling
        self._limit = limit
        self._num_actuators = num_actuators

        self.reset()

    def compute_torques(self, joint_position_errors, joint_velocities, return_tensor=False):
        with torch.no_grad():
            an_input = torch.concat([
                torch.tensor(joint_position_errors).view(-1, 1) *
                self._input_scaling[0],
                torch.tensor(joint_velocities).view(-1, 1) *
                self._input_scaling[1]
            ], axis=1).to(torch.float).to(self._device)

            torques = torch.clip(self._actuator_network.forward(
                an_input) * self._output_scaling, -self._limit, self._limit)

            if return_tensor:
                return torques

            return torques.cpu().numpy().flatten()

    def reset(self):
        self._actuator_network.reset(batch_size=self._num_actuators)
        self._actuator_network.eval()
