import numpy as np
import matplotlib.pyplot as plt
from robot import Robot

odometryFilename = "ds0_odometry_clean.dat"
groundTruthFilename = "ds0_Groundtruth_clean.dat"

importData = True # Set to True to import data from file. If set to false, a test data set will be used

# Define robot initial position and orientation. The values below are taken from the ground truth data
xInit = 1.29812900
yInit = 1.88315210
thetaInit = 2.82870000

# Intialize Robot object
robot = Robot(xInit, yInit, thetaInit)

robot.import_groundtruth(groundTruthFilename)

if importData == True:
    # Import and analyze odometry control data from specified file
    data = robot.import_odometry(odometryFilename)
    robot.run_odometry()
else:
    # Run using the test case below
    data = [[1, 0.5, 0.0, 1.0], [2, 0.0, -1/(2*np.pi), 1.0], [3, 0.5, 0.0, 1.0], [4, 0.0, 1/(2*np.pi), 1.0], [5, 0.5, 0.0, 1.0]] # [case number, linear velocity (m/s), angular velocity (rad/s), time step (sec)]
    # Move the robot according to the test control data
    print("Moving robot...")
    for case in data:
        robot.change_pos(case[1], case[2], case[3])
    print("Robot movement complete!")

# Create plot of motion 
fig, ax = plt.subplots()
robot.plot_position_GT(ax)
robot.plot_position_DR(ax)
robot.show_plot(ax)
