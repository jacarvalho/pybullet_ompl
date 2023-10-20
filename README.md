# pybullet_ompl
This repo provides interface to use OMPL for motion planning inside PyBullet. It uses OMPL python bindings.

NOTE: this library was adpated from [https://github.com/lyfkyle/pybullet_ompl](https://github.com/lyfkyle/pybullet_ompl)


![example](/images/example.gif)

# Environment
Tested with:<br>
**Python 3.8**<br>
**Ubuntu18.04**

# Installation instructions:

## Install dependencies of OMPL
https://github.com/ompl/ompl/blob/main/doc/markdown/installPyPlusPlus.md

## Install OMPL from source
It is very important that you compile ompl with the correct python version with the CMake flag.
```
git clone https://github.com/ompl/ompl.git
mkdir build/Release
cd build/Release
# cmake ../.. -DPYTHON_EXEC=/path/to/python-X.Y # This is important!!! Make sure you are pointing to the correct python version.
cmake -DCMAKE_DISABLE_FIND_PACKAGE_pypy=ON ../.. -DPYTHON_EXEC=/usr/bin/python${PYTHONV}
make -j 4 update_bindings # replace "4" with the number of cores on your machine. This step takes some time.
make -j 4 # replace "4" with the number of cores on your machine
```

## Install this library
```
pip install -e .
```

# Demo
Two examples are provided.
This demo plans the arm motion of a Franka robot.
```
python examples/demo_franka.py
```

This demo plans whole-body motion of a planar 4-link snake-like robot.
```
python examples/demo_planar.py
```

# Additional Information
1. Currently tested planners include PRM, RRT, RRTstar, RRTConnect, EST, FMT* and BIT*. But all planners in OMPL should work. Just add them in the set_planner API in PbOMPL class.
2. To work with other robot, you might need to inherit from PbOMPLRobot class in PbOMPL and override several functionalities. Refer to my_planar_robot.py for an example. Refer to demo_plannar.py for an example.
