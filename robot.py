import numpy as np
import matplotlib.pyplot as plt
import time as time

plt.style.use('_mpl-gallery')

class Robot:
    def __init__(self, xInit, yInit, thetaInit):
        # Initiallize Robot class
        self.posDeadRec = np.array([xInit, yInit, thetaInit])
        self.posGroundTruth = np.array([xInit, yInit, thetaInit, 0])
        self.odometry = np.array([0,0,0,0])

    def run_odometry(self):
        # Function to move robot according to imported odometry data
        print("Moving robot according to imported odometry...")
        for case in self.odometry:
            self.change_pos(case[1], case[2], case[3])
        print("Robot movement complete!")

    def change_pos(self, vel, omega, timeStep):
        # this fucntion calculates the new position and orientation of the robot based on the previous location and oriantationa and the command sent to the robot
        try:
            # Take the most recent calculated position for the "start" coordinates x0, y0, and intial theta
            x0 = self.posDeadRec[-1, 0]
            y0 = self.posDeadRec[-1, 1]
            theta = self.posDeadRec[-1, 2]
        except IndexError  as e:
            # If no new positions have been added yet, the intial X, Y, and theta will be used
            x0 = self.posDeadRec[0]
            y0 = self.posDeadRec[1]
            theta = self.posDeadRec[2]

        if omega == 0.0:
            # this is used in cases where angular velocity = 0, calculates new position for a linear move
            deltaX = np.cos(theta) * vel * timeStep
            deltaY = np.sin(theta) * vel * timeStep
            xNew = x0 + deltaX
            yNew = y0 + deltaY
            thetaNew = theta
            # Add the new position to the position array
            self.posDeadRec = np.vstack((self.posDeadRec, [xNew, yNew, thetaNew]))
        else:
            # this is used in cases where angular velocity =/= 0 (turning)
            radius = vel/omega # radius of rotation based on linear and angular speed
            angRot = omega * timeStep # angle change during maneuver based on angular speed
            if omega > 0: 
                # calculates the center of rotation for a LH turn
                curveCenterX = x0 + (np.cos(theta + np.pi/2) * radius)
                curveCenterY = y0 + (np.sin(theta + np.pi/2) * radius)
                thetaNewTot = theta + angRot
            elif omega < 0: 
                # calculates the center of rotation for a RH turn
                curveCenterX = x0 + (np.cos(theta + np.pi/2) * radius)
                curveCenterY = y0 + (np.sin(theta + np.pi/2) * radius)
                thetaNewTot = theta + angRot
            # Calculate a new position based off the center of rotation, the radius of rotation, and the angle of rotation
            # Use a while loop to create to generate multiple points along the curve so a smooth curve is displayed on the plot
            counter = 0
            countMax = 3
            while counter < countMax:
                angleStep = (angRot * (counter+1)) / countMax
                thetaNew = (thetaNewTot * (counter+1)) / countMax
                angStepMod = angleStep - (np.pi/2 - theta) # switches the angle change from the robot cooridnate system to the global coordinates system
                xNew = curveCenterX + (np.cos(angStepMod) * radius) 
                yNew = curveCenterY + (np.sin(angStepMod) * radius)
                # Add the new position to the position array
                self.posDeadRec = np.vstack((self.posDeadRec, [xNew, yNew, thetaNew]))
                counter += 1

    def plot_position_DR(self, ax):
        # function to plot the robot path based on dead reckoning data
        print("Plotting dead reckoning movement and ground truth position of robot...")
        xPosDR = []
        yPosDR = []
        for pos in self.posDeadRec:
            xPosDR.append(pos[0])
            yPosDR.append(pos[1])
        ax.plot(xPosDR, yPosDR, 'r-')

    def plot_position_GT(self, ax):
        # function to plot the robot path based on ground truth data
        print("Plotting dead reckoning movement and ground truth position of robot...")
        xPosGT = []
        yPosGT = []
        for pos in self.posGroundTruth:
            xPosGT.append(pos[0])
            yPosGT.append(pos[1])
        ax.plot(xPosGT, yPosGT, 'b-')
    
    def show_plot(self, ax):
        print("Generating plot!")
        ax.set_aspect( 1 )
        plt.show()

    def import_odometry(self, filename): 
        # Imports data from specified file. File should be in same directory as the python files
        print('Importing odometry data from file "' + filename + '"...')
        data = np.genfromtxt(filename)
        countMax = len(data)
        count = 0
        newData = np.array([0.0, 0.0, 0.0, 0.0])
        while count + 1 < countMax:
            # Iterate through the data and calculate time steps based on the provided time data
            t0 = data[count][0]
            t1 = data[count + 1][0]
            tStep = t1 - t0
            newPoint = np.array([count+1, data[count][1],data[count][2], tStep])
            newData = np.vstack((newData, newPoint))
            count += 1
        print("Odometry data import complete!")
        self.odometry = newData
        return newData
    
    def import_groundtruth(self, filename): 
        # Imports data from specified file. File should be in same directory as the python files
        print('Importing groundtruth data from file "' + filename + '"...')
        data = np.genfromtxt(filename)
        countMax = len(data)
        count = 0
        newData = np.array([0.0, 0.0, 0.0, 0.0])
        while count + 1 < countMax:
            # Iterate through the data and calculate time steps based on the provided time data
            t0 = data[count][0]
            t1 = data[count + 1][0]
            tStep = t1 - t0
            newPoint = np.array([data[count][1],data[count][2], data[count][3],tStep])
            newData = np.vstack((newData, newPoint))
            count += 1
        print("Groundtruth data import complete!")
        newData = np.delete(newData, 0, 0)
        self.posGroundTruth = newData
        return newData
    


        


            