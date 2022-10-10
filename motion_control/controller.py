import torch
import torch.nn as nn

import numpy as np

from common.networks import MultiLayerPerceptron


class Controller:
    def __init__(self, parameters_path, nominal_generalized_coordinates, nominal_generalized_velocities, control_decimation,
                 state_dim=48, action_dim=12, policy_layers=None, policy_activation=nn.Tanh, action_scaling=1.0):
        if policy_layers is None:
            policy_layers = [256, 256]

        self._num_layers = len(policy_layers)

        self._policy_network = MultiLayerPerceptron(
            in_dim=state_dim, out_dim=action_dim, hidden_layers=policy_layers, activation=policy_activation, dropout=0.)

        policy_state_dict = self._validate_state_dict(
            torch.load(parameters_path + '/policy.pt', map_location=torch.device('cpu'))
        )

        self._policy_network.load_state_dict(policy_state_dict)
        self._policy_network.eval()

        self._state_norm_offset = np.loadtxt(
            parameters_path + '/state_mean.txt', delimiter=',')
        self._state_norm_scaling = np.sqrt(
            np.loadtxt(parameters_path + '/state_var.txt', delimiter=','))

        self._action_scaling = action_scaling

        self._nominal_gc = np.array(nominal_generalized_coordinates).flatten()
        self._nominal_gv = np.array(nominal_generalized_velocities).flatten()

        self._desired_joint_positions = self._nominal_gc[-12:].copy()

        self._state = np.zeros(state_dim)
        self._action = np.zeros(action_dim)

        self._gravity_axis = np.array([0., 0., 1.])

        self._control_decimation = control_decimation
        self._step_callbacks = 0

    def reset(self):
        self._step_callbacks = 0
        self._desired_joint_positions = self._nominal_gc[-12:].copy()

    def step(self, base_rotation, joint_positions, base_lin_vel, base_ang_vel, joint_velocities, velocity_command):
        self._step_callbacks += 1

        if ((self._step_callbacks - 1) % self._control_decimation != 0):
            if self._step_callbacks == self._control_decimation:
                self._step_callbacks = 0

            return self._desired_joint_positions

        self._state[0:3] = base_rotation[2, :].flatten()
        self._state[3:15] = joint_positions
        self._state[15:18] = base_rotation.T @ base_ang_vel
        self._state[18:30] = joint_velocities
        self._state[30:33] = base_rotation.T @ base_lin_vel
        self._state[33:36] = velocity_command
        self._state[36:48] = self._desired_joint_positions - joint_positions

        self._state = (self._state - self._state_norm_offset) / \
            self._state_norm_scaling
        self._state = np.clip(self._state, -10., 10.)

        with torch.no_grad():
            self._action = self._policy_network.forward(
                torch.from_numpy(self._state).view(1, -1).to(torch.float)).cpu().numpy().flatten()

        self._action = np.clip(self._action, -2., 2.)

        self._desired_joint_positions = (
            self._action * self._action_scaling) + self._nominal_gc[-12:]

        return self._desired_joint_positions

    def _validate_state_dict(self, state_dict):
        valid_dict = True

        for key in state_dict.keys():
            if 'layers.' + str(self._num_layers) + '.' in key:
                valid_dict = False
                break

        if valid_dict:
            return state_dict

        mod_dict = dict()

        for key in state_dict.keys():
            mod_dict[key.replace(
                f'_fully_connected_layers.{self._num_layers}', '_output_layer')] = state_dict[key]

        return mod_dict
