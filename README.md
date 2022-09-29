# LFMC-PyVal

A Python based deployment repository for executing 
policies trained with [```LFMC-Gym```](https://github.com/ori-drs/lfmc_gym).
It can be used with both 
[PyBullet](https://pybullet.org/wordpress/) and 
[RaisimPy](https://raisim.com/sections/RaiSimPy.html).
Switching between the two engines is extremely simple.

### Prerequisites
This is **optional** and is 
only applicable if you would like to use RaisimPy. 
Ensure that you have
compiled it and that ```$PYTHONPATH``` contains the 
RaisimPy install location.
```bash
export PYTHONPATH=$PYTHONPATH:<raisim-install-directory>/lib
```

### Clone
To clone ```lfmc_pyval```, use the following command. The Python
version does not depend on other repositories.
```bash
git clone git@github.com:ori-drs/lfmc_pyval.git
```

### Install 

A ```setup.py``` script is provided with the Python
package which handles all the necessary dependencies.
It is recommended that you use a 
[virtual environment](https://docs.python.org/3/tutorial/venv.html).

If ```python3-venv``` is not already installed, use the following command.
```bash
sudo apt-get install python3-venv
```

Now create and source a virtual environment. 
```bash
cd lfmc_pyval
python3 -m venv venv
source venv/bin/activate
```

The dependencies can then be installed by
```bash
pip install -e .
```

This will install ```numpy```, ```torch```, ```pybullet```, and
```pyyaml``` along with their dependencies.

### Execute

To run the included example, just use the following command.

```bash
python scripts/command_tracking.py
```

This will execute the locomotion policy in PyBullet. To switch to
RaisimPy, simply edit the first line in the simulation
configuration file. The file is called ```simulation.yaml```
and can be found at ```$LFMC_PYVAL_PATH/configuration/simulation.yaml```.
By default, this is what Line 1 reads.
```yaml
engine: pybullet # Options: [raisim, pybullet]
```
To switch to Raisim, change it to
```yaml
engine: raisim # Options: [raisim, pybullet]
```

Assuming ```RaisimUnity``` has been launched for visualization,
you can execute the ```command_tracking.py``` script just as before.
The example will now utilize the Raisim engine.
```bash
python scripts/command_tracking.py
```

### Code Structure
    └── common                          # Utilities used by modules throughout the project
        ├── paths.py                    # Utility to handle project related paths
        ├── networks.py                 # Contains MLP and GRU classes based on PyTorch
    └── configuration                   # Configuration files
        ├── simulation.yaml             # Simulation configuration parameters
    └── motion_control                  # Locomotion controller related files
        ├── controller.py               # Controller interface class
    └── resources                       # Assets used in the project
        ├── models                      # Robot URDFs
        ├── parameters                  # Network parameters for locomotion and actuation
    └── scripts                         # Python scripts
        ├── command_tracking.py         # Example script for executing command tracking policy
    └── simulation                      # Simution utilities
        ├── actuation.py                # Contains the actuator network class for ANYmal C
        ├── pybullet.py                 # Interface utility for PyBullet
        ├── raisim.py                   # Interface utility for RaisimPy
        ├── wrapper.py                  # Wrapper to switch between PyBullet and RaisimPy
    └── setup.py                        # Installs LFMC-PyVal and dependencies

### Author(s)
[Siddhant Gangapurwala](mailto:siddhant@robots.ox.ac.uk)
