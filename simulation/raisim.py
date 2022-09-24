import numpy as np
import raisimpy as raisim

from simulation.actuation import Actuation


class RaiSimWrapper:
    def __init__(self, simulation_config, model_path_ground, model_path_robot, actuation_parameters_path):
        self._nominal_gc = np.zeros(19)
        self._nominal_gc[:3] = np.array(
            simulation_config['robot']['nominal_base_position'])

        # RaiSim quaternion format is [w, x, y, z]
        self._nominal_gc[3] = simulation_config['robot']['nominal_base_orientation'][0]
        self._nominal_gc[4] = simulation_config['robot']['nominal_base_orientation'][1]
        self._nominal_gc[5] = simulation_config['robot']['nominal_base_orientation'][2]
        self._nominal_gc[6] = simulation_config['robot']['nominal_base_orientation'][3]

        self._nominal_gc[-12:] = np.array(
            simulation_config['robot']['nominal_joint_configuration'])

        self._nominal_gv = np.zeros(18)

        self.generalized_coordinates = self._nominal_gc.copy()
        self.generalized_velocities = self._nominal_gv.copy()

        self.base_rotation = self._get_rotation_matrix_from_quaternion(
            self.generalized_coordinates[3:7])

        self._desired_joint_positions = \
            self.generalized_coordinates[-12:].copy()

        self._actuation_handler = Actuation(actuation_parameters_path)

        self._actuation_decimation = int(np.ceil(
            simulation_config['frequency']['simulation'] / simulation_config['frequency']['actuation']))
        self._step_count = 0

        self._joint_torques = np.zeros(12)
        self._robot_actuation = np.zeros(18)

        self._world = raisim.World()
        self._world.setTimeStep(
            1. / float(simulation_config['frequency']['simulation']))

        self._world.addGround()

        self._robot = self._world.addArticulatedSystem(
            model_path_robot, joint_order=simulation_config['robot']['joint_order'])

        self._robot.setState(self._nominal_gc, self._nominal_gv)
        self._robot.setPdGains(
            np.zeros(self._robot.getDOF()), np.zeros(self._robot.getDOF()))

        self._server = raisim.RaisimServer(self._world)
        self._server.launchServer(8080)

        self._server.focusOn(self._robot)

        self.update_robot_states()

    def update_robot_states(self):
        self.generalized_coordinates, self.generalized_velocities = self._robot.getState()
        self.base_rotation = self._get_rotation_matrix_from_quaternion(
            self.generalized_coordinates[3:7])

    def set_desired_joint_positions(self, desired_joint_positions):
        self._desired_joint_positions = np.array(
            desired_joint_positions).flatten()

    def apply_actuation(self, update_torques=True):
        if update_torques:
            joint_position_errors = self._desired_joint_positions - \
                                    self.generalized_coordinates[-12:]
            self._joint_torques = self._actuation_handler.compute_torques(
                joint_position_errors, self.generalized_velocities[-12:])

            self._robot_actuation[-12:] = self._joint_torques

        self._robot.setGeneralizedForce(self._robot_actuation)

    def step(self, desired_joint_positions, update_states=True):
        if update_states:
            self.update_robot_states()

        self.set_desired_joint_positions(desired_joint_positions)
        self.apply_actuation(self._step_count %
                             self._actuation_decimation == 0)

        self._world.integrate()
        self._step_count += 1

    def reset(self):
        self._robot.setState(self._nominal_gc, self._nominal_gv)
        self.update_robot_states()
        self._actuation_handler.reset()
        self._step_count = 0

    @property
    def nominal_gc(self):
        return self._nominal_gc

    @property
    def nominal_gv(self):
        return self._nominal_gv

    @staticmethod
    def _get_rotation_matrix_from_quaternion(quaternion):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

        # Extract the values from Q
        q0 = quaternion[0]
        q1 = quaternion[1]
        q2 = quaternion[2]
        q3 = quaternion[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix
