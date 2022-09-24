import numpy as np
import pybullet as p

from simulation.actuation import Actuation


class PyBulletWrapper:
    def __init__(self, simulation_config, model_path_ground, model_path_robot, actuation_parameters_path):
        self._nominal_gc = np.zeros(19)
        self._nominal_gc[:3] = np.array(
            simulation_config['robot']['nominal_base_position'])

        # PyBullet quaternion format is [x, y, z, w]
        self._nominal_gc[3] = simulation_config['robot']['nominal_base_orientation'][1]
        self._nominal_gc[4] = simulation_config['robot']['nominal_base_orientation'][2]
        self._nominal_gc[5] = simulation_config['robot']['nominal_base_orientation'][3]
        self._nominal_gc[6] = simulation_config['robot']['nominal_base_orientation'][0]

        self._nominal_gc[-12:] = np.array(
            simulation_config['robot']['nominal_joint_configuration'])

        self._nominal_gv = np.zeros(18)

        self.generalized_coordinates = self._nominal_gc.copy()
        self.generalized_velocities = self._nominal_gv.copy()

        self.base_rotation = np.array(p.getMatrixFromQuaternion(
            self.generalized_coordinates[3:7])).reshape((3, 3))

        self._desired_joint_positions = \
            self.generalized_coordinates[-12:].copy()

        self._actuation_handler = Actuation(actuation_parameters_path)

        self._joint_order = simulation_config['robot']['joint_order']
        self._joint_name_to_id = dict()

        self._joint_torques = np.zeros(12)

        self._actuation_decimation = int(np.ceil(
            simulation_config['frequency']['simulation'] / simulation_config['frequency']['actuation']))
        self._step_count = 0

        # Set up Simulation
        p.connect(
            p.GUI, options="--background_color_red=0.14 --background_color_green=0.2 --background_color_blue=0.264")

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1. /
            float(simulation_config['frequency']['simulation']),
            numSolverIterations=int(1e2),
            erp=0.2,
            contactERP=0.2,
            frictionERP=0.2
        )

        p.setGravity(0., 0., -9.81)

        self.ground = p.loadURDF(model_path_ground)

        self.robot = p.loadURDF(
            model_path_robot,
            self._nominal_gc[0:3],
            self._nominal_gc[3:7],
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_IMPLICIT_CYLINDER | p.URDF_USE_INERTIA_FROM_FILE
        )

        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)

            joint_id = info[0]
            joint_name = info[1]
            joint_type = info[2]

            if joint_type == p.JOINT_REVOLUTE:
                self._joint_name_to_id[joint_name.decode("UTF-8")] = joint_id

        for idx, joint_name in enumerate(self._joint_order):
            p.resetJointState(
                self.robot, self._joint_name_to_id[joint_name], self._nominal_gc[-12:][idx])

        self.update_robot_states()

    def update_robot_states(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.robot)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot)

        self.generalized_coordinates[:3] = np.array(base_position)
        self.generalized_velocities[:3] = np.array(base_lin_vel)

        self.generalized_coordinates[3:7] = np.array(base_orientation)
        self.generalized_velocities[3:6] = np.array(base_ang_vel)

        for idx, joint_name in enumerate(self._joint_order):
            joint_pos, joint_vel, _, _ = p.getJointState(
                self.robot, self._joint_name_to_id[joint_name])

            self.generalized_coordinates[-12:][idx] = joint_pos
            self.generalized_velocities[-12:][idx] = joint_vel

        self.base_rotation = np.array(p.getMatrixFromQuaternion(
            self.generalized_coordinates[3:7])).reshape((3, 3))

    def set_desired_joint_positions(self, desired_joint_positions):
        self._desired_joint_positions = np.array(
            desired_joint_positions).flatten()

    def apply_actuation(self, update_torques=True):
        if update_torques:
            joint_position_errors = self._desired_joint_positions - \
                self.generalized_coordinates[-12:]
            self._joint_torques = self._actuation_handler.compute_torques(
                joint_position_errors, self.generalized_velocities[-12:])

        for idx, joint_name in enumerate(self._joint_order):
            p.setJointMotorControl2(
                self.robot, self._joint_name_to_id[joint_name], p.VELOCITY_CONTROL, targetVelocity=0., force=0.)
            p.setJointMotorControl2(
                self.robot, self._joint_name_to_id[joint_name], p.TORQUE_CONTROL,
                force=self._joint_torques[idx])

    def step(self, desired_joint_positions, update_states=True):
        if update_states:
            self.update_robot_states()

        self.set_desired_joint_positions(desired_joint_positions)
        self.apply_actuation(self._step_count %
                             self._actuation_decimation == 0)

        p.stepSimulation()
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=self.generalized_coordinates[:3]
        )

        self._step_count += 1

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.robot, self._nominal_gc[:3], self._nominal_gc[3:7])
        p.resetBaseVelocity(
            self.robot, self._nominal_gv[:3], self._nominal_gv[3:6])

        for idx, joint_name in enumerate(self._joint_order):
            p.resetJointState(
                self.robot, self._joint_name_to_id[joint_name], self._nominal_gc[-12:][idx], 0.)

        self.update_robot_states()
        self._actuation_handler.reset()
        self._step_count = 0

    @property
    def nominal_gc(self):
        return self._nominal_gc

    @property
    def nominal_gv(self):
        return self._nominal_gv
