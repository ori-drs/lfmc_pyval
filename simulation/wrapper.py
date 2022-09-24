class Simulation:
    def __init__(self, simulation_config, model_path_ground, model_path_robot, actuation_parameters_path):
        engine = str(simulation_config['engine'])

        # Set up a wrapper
        if engine == 'pybullet':
            from simulation.pybullet import PyBulletWrapper
            self._wrapper = PyBulletWrapper(
                simulation_config, model_path_ground, model_path_robot, actuation_parameters_path)

        elif engine == 'raisim':
            from simulation.raisim import RaiSimWrapper
            self._wrapper = RaiSimWrapper(
                simulation_config, model_path_ground, model_path_robot, actuation_parameters_path)

        else:
            raise ValueError(f'Engine {engine} is not Supported. Pick between pybullet or raisim.')

    def update_robot_states(self):
        self._wrapper.update_robot_states()
    
    def set_desired_joint_positions(self, desired_joint_positions):
        self._wrapper.set_desired_joint_positions(desired_joint_positions)
    
    def apply_actuation(self, update_torques=True):
        self._wrapper.apply_actuation(update_torques)

    def step(self, desired_joint_positions, update_states=True):
        self._wrapper.step(desired_joint_positions, update_states)

    def reset(self):
        self._wrapper.reset()

    @property
    def nominal_gc(self):
        return self._wrapper.nominal_gc

    @property
    def nominal_gv(self):
        return self._wrapper.nominal_gv

    @property
    def generalized_coordinates(self):
        return self._wrapper.generalized_coordinates

    @property
    def generalized_velocities(self):
        return self._wrapper.generalized_velocities

    @property
    def base_rotation(self):
        return self._wrapper.base_rotation
