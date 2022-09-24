import time
import yaml

import numpy as np

from common.paths import ProjectPaths
from simulation.wrapper import Simulation
from motion_control.controller import Controller


def main():
    paths = ProjectPaths()

    config_file = open(paths.CONFIGURATION_PATH + '/simulation.yaml')
    simulation_config = yaml.safe_load(config_file)

    simulation_frequency = simulation_config['frequency']['simulation']
    control_frequency = simulation_config['frequency']['motion_control']

    sim = Simulation(
        simulation_config=simulation_config,
        model_path_ground=paths.MODELS_PATH + '/plane/plane.urdf',
        model_path_robot=paths.MODELS_PATH + '/anymal_c/urdf/model.urdf',
        actuation_parameters_path=paths.PARAMETERS_PATH + '/actuation/anymal_c.pt'
    )

    controller = Controller(
        parameters_path=paths.PARAMETERS_PATH +
        '/controller/' + str(control_frequency),
        nominal_generalized_coordinates=sim.nominal_gc,
        nominal_generalized_velocities=sim.nominal_gv,
        control_decimation=int(simulation_frequency / control_frequency)
    )

    # Internally calls update_state()
    sim.reset()
    
    velocity_command = np.random.rand(3) * 2. - 1.

    elapsed_time = 0
    elapsed_steps = 0

    while elapsed_time < 30:
        pre_step = time.time()

        # All quantities are in world frame.
        # Note, on some robots, angular velocity is defined in base frame.
        # Controller step is called at 200 Hz.
        # The control-decimation parameter internally takes care of the right control frequency
        desired_joint_positions = controller.step(
            base_rotation=sim.base_rotation,
            joint_positions=sim.generalized_coordinates[-12:],
            base_lin_vel=sim.generalized_velocities[:3],
            base_ang_vel=sim.generalized_velocities[3:6],
            joint_velocities=sim.generalized_velocities[-12:],
            velocity_command=velocity_command
        )

        # Step through the simulator - actuates the joints and integrates the sim
        sim.step(desired_joint_positions)

        # To render in real-time
        delay = time.time() - pre_step
        time.sleep(max(1. / simulation_frequency - delay, 0.))

        elapsed_steps += 1

        # Update velocity command
        if elapsed_steps % (simulation_frequency * 3) == 0:
            velocity_command = np.random.rand(3) * 2. - 1.

            if np.linalg.norm(velocity_command) < 0.35:
                velocity_command *= 0.

        elapsed_time += time.time() - pre_step


if __name__ == '__main__':
    main()
