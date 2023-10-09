import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from map import Map

# Define filenames and test data sets
odometryFilename = "ds0_Odometry.dat"
groundTruthFilename = "ds0_Groundtruth.dat"
landmarkGTFilename = "ds0_Landmark_Groundtruth.dat"
measurementFilename = "ds0_Measurement.dat"
barcodeFilename = "ds0_Barcodes.dat"
odometryTest = [[0.5, 0.0, 1.0], [0.0, -1/(2*np.pi), 1.0], [0.5, 0.0, 1.0], [0.0, 1/(2*np.pi), 1.0], [0.5, 0.0, 1.0]] # [linear velocity (m/s), angular velocity (rad/s), time step (sec)]
measurementTest = [[2,3,0,6], [0,3,0,13], [1,-2,0,17]] # [X, Y, theta, landmark barcode #]

# Define robot initial position and orientation. The values below are taken from the ground truth data
xInit = 1.29812900
yInit = 1.88315210
thetaInit = 2.82870000

# Intialize Robot and Map object
robot = Robot(xInit, yInit, thetaInit)

# (Question 2)
# Move the robot according to the test control data
print("QUESTION 2:")
robot.run_test_odometry(odometryTest)
print("Plot will be displayed after all calculations finish.")

# Create plot
fig, ax = plt.subplots(figsize=(8, 8))
robot.plot_position_DR(ax)
ax.set_aspect(1)
ax.set(xlabel='X Position (m)', ylabel='Y Position (m)', title='Robot Dead Reckoning Position, Test Data (Question 2)')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
ax.legend()

# Reset robot odometry and position data
robot.reset_odometry_pos(xInit, yInit, thetaInit)


# (Question 3)
print("")
print("QUESTION 3:")
# Import robot and landmark groundtruth data
robot.import_landmark_GT(landmarkGTFilename)
robot.import_groundtruth(groundTruthFilename)

# Import and analyze odometry control data from specified file
data = robot.import_odometry(odometryFilename)
robot.run_odometry()
print("Plot will be displayed after all calculations finish.")

# Create plot of motion 
fig, bx = plt.subplots(figsize=(8,8))
robot.plot_position_GT(bx)
robot.plot_position_DR(bx)
robot.plot_landmark_pos(bx)
bx.set_aspect(1)
bx.set(xlabel='X Position (m)', ylabel='Y Position (m)', title='Robot Dead Reckoning and Ground Truth Position, Imported Data (Question 3)')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
bx.legend()


# (Question 6)
print("")
print("QUESTION 6:")
# Find landmark heading and distance based on the given robot positions
caseNum = 1
for case in measurementTest:
    [xPos, yPos] = robot.landmark_pos(case[3])
    [dist, phiRob] = robot.find_point_distance_heading(xPos,yPos,case[0],case[1],case[2])
    print("Robot position, Case " + str(caseNum) + " [X(m), Y(m), Theta(rad)]:")
    print([case[0],case[1],case[2]])
    print("Distance and Heading to Landmark, Case " + str(caseNum) + "[Distance (m), Heading (rad, relative to robot)]:")
    print([dist, phiRob])
    caseNum += 1


# (Question 8)
print("")
print("QUESTION 8:")
# Import robot and measurement data
robot.import_measurement(measurementFilename)
robot.import_landmark_barcodes(barcodeFilename)

# Run odometry control data from specified file using particle filter and import measurement data
robot.run_odometry_particle_fliter()
print("Plot will be displayed after all calculations finish.")

# Create plot of motion 
fig, cx = plt.subplots(figsize=(8,8))
robot.plot_position_GT(cx)
robot.plot_position_PF(cx)
robot.plot_landmark_pos(cx)
robot.plot_measurement_pos(cx)
cx.set_aspect(1)
cx.set(xlabel='X Position (m)', ylabel='Y Position (m)', title='Robot Position Estimate Using Particle Filter and Imported Ground Truth Position (Question 8)')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
cx.legend()

# Plot all graphs
print("Generating plots!")
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
plt.show()