import numpy as np
import matplotlib.pyplot as plt
import time as time
import math as math

plt.style.use('_mpl-gallery')

class Robot:
    def __init__(self, xInit, yInit, thetaInit):
        # Initiallize Robot class
        self.posDeadRec = np.array([xInit, yInit, thetaInit])
        self.posGroundTruth = np.array([xInit, yInit, thetaInit, 0, 0])
        self.posEstPF = np.array([xInit, yInit, thetaInit])
        self.landmarkGT = {}
        self.landmarkBarcodes = {}
        self.odometry = np.array([0,0,0,0])
        self.measurement = np.array([0,0,0,0,0])
        self.belState = np.array([])
        self.measurementPoints = np.array([0,0,0])

    def run_test_odometry(self, odom):
        # Function to move robot according to give odometry data
        # Data given in format [[v1, w1, t_step1], [v2, w2, t_step2], etc...]
        print("Moving robot according to provided test odometry...")
        for command in odom:
            newPos = self.change_pos(command[0], command[1], command[2], self.posDeadRec)
            # Add the new position to the position array
            self.posDeadRec = np.vstack((self.posDeadRec, newPos))
        print("Robot movement complete!")

    def run_odometry(self):
        # Function to move robot according to imported odometry data
        print("Moving robot according to imported odometry...")
        for command in self.odometry:
            newPos = self.change_pos(command[0], command[1], command[2], self.posDeadRec)
            # Add the new position to the position array
            self.posDeadRec = np.vstack((self.posDeadRec, newPos))
        print("Robot movement complete!")

    def run_odometry_particle_fliter(self):
        # Function to move robot according to imported odometry data

        ### PARTICLE FILTER TUNING PARAMETERS ##################################
        minParticles = 50                  # The minimum number of particles that will be generated
        maxParticles = 400                  # The maximum number of particles that will be generated
        initialParticleCount = 200          # The initial number of particles that will be generated

        maxParticleSpread = 0.15           # The maximum x/y distance that a particle will be generated from the previous particle
        minParticleSpread = 0.005           # The minimum x/y distance that a particle will be generated from the previous particle
        maxParticleTurn = np.pi/16          # The maximum difference in theta angle between a new particle and the previous particle
        minParticleTurn = np.pi/64          # The minimum difference in theta angle between a new particle and the previous particle

        initialMaxParticleSpread = 0.25     # The maximum x/y distance that a particle will be generated from the intial position
        initialMaxParticleTurn = np.pi/8    # The maximum difference in theta angle between a new particle and the intial theta angle
                
        sigmaMin = 0.1                     # The minimum "sigma" for the position measured by the sensor
        sigmaMinRange = 0.5                # The distance from the sensor where sigma begins to increase (representing an increase in uncertainty)
        sigmaIncreaseRate = 0.1             # The rate that sigma increases as the waypoint moves further from the sensor
        ########################################################################

        print("Moving robot according to imported odometry using particle filter to estimate position...")
        measureIndex = 0
        newMeasurement = True
        particlesGenerated = False
        for command in self.odometry:
            timeOdom = command[3]
            try:
                nextMeasurment = self.measurement[measureIndex]
                newMeasurement = True
                subject = self.landmarkBarcodes[int(nextMeasurment[0])]
            except IndexError as e:
                nextMeasurment = [0,0,0,0]
                newMeasurement = False
                subject = 0
            if (timeOdom > nextMeasurment[3]) and (subject > 5) and (newMeasurement == True):
                if particlesGenerated == False:
                    initialPosX = self.posEstPF[-1][0]
                    initialPosY = self.posEstPF[-1][1]
                    initialPosTheta = self.posEstPF[-1][2]
                    print(self.posEstPF)
                    print(initialPosX)
                    initialParticlesX = ((np.random.rand(initialParticleCount, 1) - 0.5) * initialMaxParticleSpread * 2) + initialPosX
                    initialParticlesY = ((np.random.rand(initialParticleCount, 1)- 0.5) * initialMaxParticleSpread * 2) + initialPosY
                    initialParticlesTheta = ((np.random.rand(initialParticleCount, 1) - 0.5) * initialMaxParticleTurn * 2)  + initialPosTheta
                    initialParticlesProbDist = (np.random.rand(initialParticleCount, 1) * 0 + 1)
                    self.belState = np.hstack((initialParticlesX, initialParticlesY, initialParticlesTheta, initialParticlesProbDist))
                    particlesGenerated = True
                landmarkPos = self.landmark_pos(subject)
                totalProb = 0
                highProbParticleArray = np.array([[0,0,0,0],[0,0,0,0]])
                minHighProb = 0
                highProbPointCount = 20
                for particle in self.belState:
                    while particle[2] > np.pi*2:
                        particle[2] += -np.pi*2
                    while particle[2] < 0:
                        particle[2] += np.pi*2
                    measureDist = nextMeasurment[1]
                    measurePhi = nextMeasurment[2]
                    particleDist, particlePhi = self.find_point_distance_heading(landmarkPos[0], landmarkPos[1], particle[0], particle[1], particle[2])
                    xEstLandmarkParticle = particleDist * np.cos(particlePhi)
                    yEstLandmarkParticle = particleDist * np.sin(particlePhi)
                    xEstLandmarkMeasurement = measureDist * np.cos(measurePhi)
                    yEstLandmarkMeasurement = measureDist * np.sin(measurePhi)
                    magEstError = ((xEstLandmarkParticle - xEstLandmarkMeasurement)**2 + (yEstLandmarkParticle - yEstLandmarkMeasurement)**2)**0.5
                    if measureDist >  sigmaMinRange:
                        sigmaAdjusted = (sigmaIncreaseRate * (measureDist - sigmaMinRange)) + sigmaMin
                    else:
                        sigmaAdjusted = sigmaMin
                    zError = magEstError / sigmaAdjusted
                    particleProb = 2 * ((1 / 2) * np.exp(-zError**2 / 2))
                    particle[3] = particleProb
                    totalProb += particleProb
                    highProbMinList = np.amin(highProbParticleArray, axis=0)
                    highProbMinIndex = int(highProbMinList[3])
                    minHighProb = highProbParticleArray[highProbMinIndex]
                    if (minHighProb[3] < particleProb) and (np.size(highProbParticleArray, axis=0) >= highProbPointCount):
                        highProbParticleArray[highProbMinIndex] = particle
                    elif (minHighProb[3] < particleProb) and (np.size(highProbParticleArray, axis=0) < highProbPointCount):
                        highProbParticleArray = np.vstack([highProbParticleArray, particle])
                highProbParticle = np.mean(highProbParticleArray, axis=0)
                highProb = float(highProbParticle[3])
                newPosEst = highProbParticle
                newPosEst = np.delete(newPosEst, 3, axis=0)
                #print()
                #print(highProbParticle)
                self.posEstPF = np.vstack((self.posEstPF, newPosEst))
                self.measurementPoints =  np.vstack((self.measurementPoints, newPosEst))
                newParticleCount = int((1 - highProb) * (maxParticles - minParticles)) + minParticles
                print(newParticleCount)
                newParticleArray = np.array([[0,0,0,0]])
                #time.sleep(10)
                for particle in range(newParticleCount):
                    randomSelect = totalProb * np.random.rand()
                    counter = -1
                    while randomSelect >= 0:
                        counter+= 1
                        randomParticle = self.belState[counter]
                        randomSelect += -(randomParticle[3])
                    newParticleSpread = ((maxParticleSpread - minParticleSpread) * (1 - randomParticle[3])) + minParticleSpread
                    newParticleTurn = ((maxParticleTurn - minParticleTurn) * (1 - randomParticle[3])) + minParticleTurn
                    newParticle = np.array([((np.random.rand() - 0.5) * newParticleSpread * 2) + randomParticle[0]])
                    newParticle = np.hstack([newParticle, ((np.random.rand() - 0.25) * newParticleSpread) * 2 + randomParticle[1]])
                    newParticle = np.hstack([newParticle, ((np.random.rand() - 0.25) * newParticleTurn * 2)  + randomParticle[2]])
                    newParticle = np.hstack([newParticle, 1])
                    newParticleArray = np.vstack((newParticleArray, newParticle))
                newParticleArray = np.delete(newParticleArray, 0, axis=0)
                #print(newParticleArray)
                #print()
                #time.sleep(2)
                self.belState = newParticleArray
                measureIndex += 1
            elif subject <= 5 and newMeasurement == True:
                measureIndex += 1
            newPos = self.change_pos(command[0], command[1], command[2], self.posEstPF)
            for pos in self.belState:
                pos[0], pos[1], pos[2] = self.change_pos(command[0], command[1], command[2], pos)
            # Add the new position to the position array
            self.posEstPF = np.vstack((self.posEstPF, newPos))
        print("Robot movement complete!")

    def change_pos(self, vel, omega, timeStep, pos):
        # this fucntion calculates the new position and orientation of the robot based on the previous location and oriantationa and the command sent to the robot
        try:
            # Take the most recent calculated position for the "start" coordinates x0, y0, and intial theta
            x0 = pos[-1, 0]
            y0 = pos[-1, 1]
            theta = pos[-1, 2]
        except IndexError as e:
            # If no new positions have been added yet, the intial X, Y, and theta will be used
            x0 = pos[0]
            y0 = pos[1]
            theta = pos[2]
        if omega == 0.0:
            # this is used in cases where angular velocity = 0, calculates new position for a linear move
            deltaX = np.cos(theta) * vel * timeStep
            deltaY = np.sin(theta) * vel * timeStep
            xNew = x0 + deltaX
            yNew = y0 + deltaY
            thetaNew = theta
            return np.array([xNew, yNew, thetaNew])
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
            curveArray = np.array([0,0,0])
            while counter < countMax:
                angleStep = (angRot * (counter+1)) / countMax
                thetaNew = (thetaNewTot * (counter+1)) / countMax
                angStepMod = angleStep - (np.pi/2 - theta) # switches the angle change from the robot cooridnate system to the global coordinates system
                xNew = curveCenterX + (np.cos(angStepMod) * radius) 
                yNew = curveCenterY + (np.sin(angStepMod) * radius)
                curveArray = np.vstack([curveArray, np.array([xNew, yNew, thetaNew])])
                counter += 1
            curveArray = np.delete(curveArray, 0, axis=0)
            return curveArray[countMax-1]

    def plot_position_DR(self, ax):
        # function to plot the robot path based on dead reckoning data
        xPosDR = []
        yPosDR = []
        for pos in self.posDeadRec:
            xPosDR.append(pos[0])
            yPosDR.append(pos[1])
        ax.plot(xPosDR, yPosDR, 'r-', label=' Robot Dead Reckoning Position')

    def plot_position_PF(self, ax):
        # function to plot the robot path based on dead reckoning data
        xPosPF = []
        yPosPF = []
        for pos in self.posEstPF:
            xPosPF.append(pos[0])
            yPosPF.append(pos[1])
        ax.plot(xPosPF, yPosPF, 'g-', label=' Robot Position Estimate Using Particle Filter')

    def plot_position_GT(self, ax):
        # function to plot the robot path based on ground truth data
        xPosGT = []
        yPosGT = []
        for pos in self.posGroundTruth:
            xPosGT.append(pos[0])
            yPosGT.append(pos[1])
        ax.plot(xPosGT, yPosGT, 'b-', label='Robot Ground Truth Position')

    def plot_landmark_pos(self, ax):
        # Plots the locations on the landmarks
        xPos = []
        yPos = []
        for pos in self.landmarkGT:
            xPos.append(float(self.landmarkGT[pos][0]))
            yPos.append(float(self.landmarkGT[pos][1]))
        ax.scatter(xPos, yPos, c='k', label='Landmark Ground Truth Position')

    def plot_measurement_pos(self, ax):
        # Plots the locations on the landmarks
        xPos = []
        yPos = []
        for pos in self.measurementPoints:
            xPos.append(float(pos[0]))
            yPos.append(float(pos[1]))
        ax.scatter(xPos, yPos, c='k', label='Landmark Ground Truth Position')

    def import_odometry(self, filename): 
        # Imports data from specified odometry file. File should be in same directory as the python files
        print('Importing odometry data from file "' + filename + '"...')
        data = np.genfromtxt(filename, skip_header=4)
        countMax = len(data)
        count = 0
        newData = np.array([0.0, 0.0, 0.0, 0.0])
        while count + 1 < countMax:
            # Iterate through the data and calculate time steps based on the provided time data
            t0 = data[count][0]
            t1 = data[count + 1][0]
            tStep = t1 - t0
            newPoint = np.array([data[count][1],data[count][2], tStep, data[count][0]])
            newData = np.vstack((newData, newPoint))
            count += 1
        print("Odometry data import complete!")
        self.odometry = newData
        return newData
    
    def import_groundtruth(self, filename): 
        # Imports data from specified robot groundtruth file. File should be in same directory as the python files
        print('Importing groundtruth data from file "' + filename + '"...')
        data = np.genfromtxt(filename, skip_header=4)
        countMax = len(data)
        count = 0
        newData = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        while count + 1 < countMax:
            # Iterate through the data and calculate time steps based on the provided time data
            t0 = data[count][0]
            t1 = data[count + 1][0]
            tStep = t1 - t0
            newPoint = np.array([data[count][1],data[count][2], data[count][3],tStep, data[count][0]])
            newData = np.vstack((newData, newPoint))
            count += 1
        print("Groundtruth data import complete!")
        newData = np.delete(newData, 0, 0)
        self.posGroundTruth = newData
        return newData

    def import_measurement(self, filename): 
        # Imports data from specified robot groundtruth file. File should be in same directory as the python files
        print('Importing measurement data from file "' + filename + '"...')
        data = np.genfromtxt(filename, skip_header=4)
        countMax = len(data)
        count = 0
        newData = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        while count < countMax:
            # Iterate through the data and calculate time steps based on the provided time data
            try:
                t0 = data[count-1][0]
                t1 = data[count][0]
                tStep = t1 - t0
            except IndexError as e:
                tStep = 0
            newPoint = np.array([data[count][1],data[count][2], data[count][3], data[count][0], tStep])
            newData = np.vstack((newData, newPoint))
            count += 1
        print("Measurement data import complete!")
        newData = np.delete(newData, 0, axis=0)
        self.measurement = newData
        return newData

    def import_landmark_GT(self, filename): 
        # Imports data from specified file. File should be in same directory as the python files
        print('Importing landmark groundtruth data from file "' + filename + '"...')
        data = np.genfromtxt(filename, skip_header=4)
        countMax = len(data)
        count = 0
        newData = {}
        while count < countMax:
            newData[int(data[count][0])] = [data[count][1],data[count][2]]
            count += 1
        print("Landmark groundtruth data import complete!")
        self.landmarkGT = newData
        return newData
    
    def import_landmark_barcodes(self, filename): 
        # Imports data from specified file. File should be in same directory as the python files
        print('Importing landmark groundtruth data from file "' + filename + '"...')
        data = np.genfromtxt(filename, skip_header=4)
        countMax = len(data)
        count = 0
        newData = {}
        while count < countMax:
            newData[int(data[count][1])] = int(data[count][0])
            count += 1
        print("Landmark barcode data import complete!")
        self.landmarkBarcodes = newData
        return newData

    def landmark_pos(self, barcode):
        # Returns the x,y coordinates for a landmark based on the provided barcode number
        [xPos, yPos] = self.landmarkGT[barcode]
        return [xPos, yPos]
    
    def find_point_distance_heading(self, pointX, pointY, robX=0, robY=0, robTheta=0):
        #function to find the heading from the robot to a given point (typically this will be a landmark)
        dist = math.dist([robX, robY], [pointX, pointY])
        phiGlobal = np.arctan2((pointY-robY),(pointX - robX))
        phiRob = phiGlobal - robTheta
        while phiRob < 0:
            phiRob += np.pi*2
        return dist, phiRob
    
    def reset_odometry_pos(self, xInit, yInit, thetaInit):
        # Function to clear stored odometry and position data
        self.odometry = np.array([0,0,0,0])
        self.posDeadRec = np.array([xInit, yInit, thetaInit])
    


        


            