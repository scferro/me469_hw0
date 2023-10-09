import numpy as np
import matplotlib.pyplot as plt
import time as time
import math as math

plt.style.use('_mpl-gallery')

class Robot:
    def __init__(self, xInit, yInit, thetaInit):
        # Initiallize the Robot class
        self.posDeadRec = np.array([xInit, yInit, thetaInit, 0])
        self.posGroundTruth = np.array([xInit, yInit, thetaInit, 0, 0])
        self.posEstPF = np.array([xInit, yInit, thetaInit, 0])
        self.landmarkGT = {}
        self.landmarkBarcodes = {}
        self.odometry = np.array([0,0,0,0])
        self.measurement = np.array([0,0,0,0,0])
        self.belState = np.array([])
        self.measurementPoints = np.array([0,0,0,0])
        self.errorDR = np.array([0,0,0])
        self.errorPF = np.array([0,0,0])

    def run_test_odometry(self, odom):
        # Function to move robot according to given odometry data
        # Odom data given in format [[v1, w1, t_step1], [v2, w2, t_step2], etc...]
        print("Moving robot according to provided test odometry...")
        # Iterate through the odometry commands
        time = 0
        for command in odom:
            time += command[2]
            newPos = self.change_pos(command[0], command[1], command[2], self.posDeadRec)
            newPosTime = np.array([newPos[0], newPos[1], newPos[2], time])
            # Add the new position to the position array
            self.posDeadRec = np.vstack((self.posDeadRec, newPosTime))
        print("Robot movement complete!")

    def run_odometry(self):
        # Function to move robot according to imported odometry data
        print("Moving robot according to imported odometry...")
        # Iterate through the odometry commands
        for command in self.odometry:
            timeOdom = command[3]
            newPos = self.change_pos(command[0], command[1], command[2], self.posDeadRec)
            newPosTime = np.array([newPos[0], newPos[1], newPos[2], timeOdom])
            # Add the new position to the position array
            self.posDeadRec = np.vstack((self.posDeadRec, newPosTime))
        print("Robot movement complete!")

    def run_odometry_particle_fliter(self):
        # Function to move robot according to imported odometry data, using particle filter and measurment data to correct for error

        ### PARTICLE FILTER TUNING PARAMETERS ##################################
        minParticles = 100                  # The minimum number of particles that will be generated
        maxParticles = 500                  # The maximum number of particles that will be generated
        initialParticleCount = 200          # The initial number of particles that will be generated

        maxParticleSpread = 0.15            # The maximum x/y distance that a particle will be generated from the previous particle
        maxParticleTurn = np.pi/8           # The maximum difference in theta angle between a new particle and the previous particle

        initialMaxParticleSpread = 0.25     # The maximum x/y distance that a particle will be generated from the intial position
        initialMaxParticleTurn = np.pi/8    # The maximum difference in theta angle between a new particle and the intial theta angle
                
        sigmaMin = 0.1                      # The minimum "sigma" for the position measured by the sensor
        sigmaMinRange = 1                   # The distance from the sensor where sigma begins to increase (representing an increase in uncertainty)
        sigmaIncreaseRate = 0.0             # The rate that sigma increases as the waypoint moves further from the sensor

        highProbPointCount = 30             # The number of high probability particles to use when calculating an estimate of the current state
        ########################################################################

        print("Moving robot according to imported odometry using particle filter to estimate position...")
        measureIndex = 0
        newMeasurement = True
        particlesGenerated = False
        # Iterate through the odometry commands
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
            # Check if the robot has reached a new measurement point based on the time
            # If the robot has recieved a new measurement, update the belief state and current position estimate using the new measurement
            if (timeOdom > nextMeasurment[3]) and (subject > 5) and (newMeasurement == True):
                landmarkPos = self.landmark_pos(subject)
                totalProb = 0
                highProbParticleArray = np.array([[0,0,0,0],[0,0,0,0]])
                minHighProb = 0
                # If particles have not been generated yet (first measurement), generate the initial particle set
                if particlesGenerated == False:
                    initialPosX = self.posEstPF[-1][0]
                    initialPosY = self.posEstPF[-1][1]
                    initialPosTheta = self.posEstPF[-1][2]
                    initialParticlesX = ((np.random.rand(initialParticleCount, 1) - 0.5) * initialMaxParticleSpread * 2) + initialPosX
                    initialParticlesY = ((np.random.rand(initialParticleCount, 1)- 0.5) * initialMaxParticleSpread * 2) + initialPosY
                    initialParticlesTheta = ((np.random.rand(initialParticleCount, 1) - 0.5) * initialMaxParticleTurn * 2)  + initialPosTheta
                    initialParticlesProbDist = (np.random.rand(initialParticleCount, 1) * 0 + 1)
                    self.belState = np.hstack((initialParticlesX, initialParticlesY, initialParticlesTheta, initialParticlesProbDist))
                    particlesGenerated = True
                # Iterate throught the particle set and calculate the importance factor/probability for each particle
                for particle in self.belState:
                    # Adjust particle angle to be bewteen 0 and 2pi
                    while particle[2] > np.pi*2:
                        particle[2] += -np.pi*2
                    while particle[2] < 0:
                        particle[2] += np.pi*2
                    # Calculate the distance between the measured position of the landmark relative to the robot and the estimated position relative to the particle
                    # This distance can be considered the estimate "error" and will be used to calculate the importance factor/probability for the particle
                    measureDist = nextMeasurment[1]
                    measurePhi = nextMeasurment[2]
                    particleDist, particlePhi = self.find_point_distance_heading(landmarkPos[0], landmarkPos[1], particle[0], particle[1], particle[2])
                    xEstLandmarkParticle = particleDist * np.cos(particlePhi)
                    yEstLandmarkParticle = particleDist * np.sin(particlePhi)
                    xEstLandmarkMeasurement = measureDist * np.cos(measurePhi)
                    yEstLandmarkMeasurement = measureDist * np.sin(measurePhi)
                    magEstError = ((xEstLandmarkParticle - xEstLandmarkMeasurement)**2 + (yEstLandmarkParticle - yEstLandmarkMeasurement)**2)**0.5
                    # Calculate measurement uncertainty based on input parameters and the measurement distance
                    if measureDist >  sigmaMinRange:
                        sigma = (sigmaIncreaseRate * (measureDist - sigmaMinRange)) + sigmaMin
                    else:
                        sigma = sigmaMin
                    # Use the calculated uncertainty to calculate the importance factor/probability for the particle
                    zError = magEstError / sigma
                    particleProb = 2 * ((1 / 2) * np.exp(-zError**2 / 2))
                    particle[3] = particleProb
                    totalProb += particleProb
                    # If the new particle is one of the highest probabilty particles in the particle set, add it to the high probability particle array
                    highProbMinList = np.amin(highProbParticleArray, axis=0)
                    highProbMinIndex = int(highProbMinList[3])
                    minHighProb = highProbParticleArray[highProbMinIndex]
                    if (minHighProb[3] < particleProb) and (np.size(highProbParticleArray, axis=0) >= highProbPointCount):
                        highProbParticleArray[highProbMinIndex] = particle
                    elif (minHighProb[3] < particleProb) and (np.size(highProbParticleArray, axis=0) < highProbPointCount):
                        highProbParticleArray = np.vstack([highProbParticleArray, particle])
                # Average the high probability particle array to find a single position estimate and add that estimate to the position estimate array
                highProbParticle = np.mean(highProbParticleArray, axis=0)
                highProb = float(highProbParticle[3])
                newPosEst = highProbParticle
                newPosEst[3] = timeOdom
                self.posEstPF = np.vstack((self.posEstPF, newPosEst))
                self.measurementPoints =  np.vstack((self.measurementPoints, newPosEst))
                # Calculate the new number of particles to be generated based on the calculated probability of the position estimate
                newParticleCount = int((1 - highProb) * (maxParticles - minParticles)) + minParticles
                newParticleArray = np.array([[0,0,0,0]])
                # Generate new particles
                for particle in range(newParticleCount):
                    # Select a random particle to use as a starting point. Particles are weighted based on importance factor/probability
                    randomSelect = totalProb * np.random.rand()
                    counter = -1
                    while randomSelect >= 0:
                        counter+= 1
                        randomParticle = self.belState[counter]
                        randomSelect += -(randomParticle[3])
                    # Calculate the maximum spread and turn for the new particle based on the importance factor/probability of the randomly selected particle
                    newParticleSpread = (maxParticleSpread * (1 - randomParticle[3]))
                    newParticleTurn = (maxParticleTurn * (1 - randomParticle[3]))
                    # Generate the new particle and add it to the new particle array
                    newParticle = np.array([((np.random.rand() - 0.5) * newParticleSpread * 2) + randomParticle[0]])
                    newParticle = np.hstack([newParticle, ((np.random.rand() - 0.25) * newParticleSpread) * 2 + randomParticle[1]])
                    newParticle = np.hstack([newParticle, ((np.random.rand() - 0.25) * newParticleTurn * 2)  + randomParticle[2]])
                    newParticle = np.hstack([newParticle, 1])
                    newParticleArray = np.vstack((newParticleArray, newParticle))
                # Update the belief state with the array of new particles
                self.belState = newParticleArray
                measureIndex += 1
            elif subject <= 5 and newMeasurement == True:
                # Subjects 1-5 are other mobile robots and are not used for measurement calculations. These measurements are skipped.
                measureIndex += 1
                # Calculate the new position estimate based on odometry data and add it to the to the position estimate array
                newPosEst = self.change_pos(command[0], command[1], command[2], self.posEstPF)
                newPosEstTime = np.array([newPosEst[0], newPosEst[1], newPosEst[2], timeOdom])
                self.posEstPF = np.vstack((self.posEstPF, newPosEstTime))
            else:
                # If no measurement, calculate the new position estimate based on odometry data and add it to the to the position estimate array
                newPosEst = self.change_pos(command[0], command[1], command[2], self.posEstPF)
                newPosEstTime = np.array([newPosEst[0], newPosEst[1], newPosEst[2], timeOdom])
                self.posEstPF = np.vstack((self.posEstPF, newPosEstTime))
            # Update the position of every particle in the belief state based on odometry data
            for pos in self.belState:
                pos[0], pos[1], pos[2] = self.change_pos(command[0], command[1], command[2], pos)
        self.measurementPoints = np.delete(self.measurementPoints, 0, axis=0)    
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
            # This is used in cases where angular velocity = 0, calculates new position for a linear move based on current heading, linear velocity, and time step
            deltaX = np.cos(theta) * vel * timeStep
            deltaY = np.sin(theta) * vel * timeStep
            xNew = x0 + deltaX
            yNew = y0 + deltaY
            thetaNew = theta
            return np.array([xNew, yNew, thetaNew])
        else:
            # This is used in cases where angular velocity =/= 0 (turning), calculates new position for a linear move based on current heading, linear and angularvelocity, and time step
            radius = abs(vel/omega) # radius of rotation based on linear and angular speed
            angRot = omega * timeStep # angle change during maneuver based on angular speed
            angRotMod = angRot - (np.pi/2 - theta) # convert angle change to global coordinate system
            if omega > 0: 
                # Calculates the center of rotation for a LH turn
                curveCenterX = x0 + (np.cos(theta + np.pi/2) * radius)
                curveCenterY = y0 + (np.sin(theta + np.pi/2) * radius)
                thetaNew = theta + angRot
                # Calculate a new position based off the center of rotation, the radius of rotation, and the angle of rotation
                xNew = curveCenterX + (np.cos(angRotMod) * radius) 
                yNew = curveCenterY + (np.sin(angRotMod) * radius)
            elif omega < 0: 
                # Calculates the center of rotation for a RH turn
                curveCenterX = x0 + (np.cos(theta - np.pi/2) * radius)
                curveCenterY = y0 + (np.sin(theta - np.pi/2) * radius)
                thetaNew = theta + angRot
                # Calculate a new position based off the center of rotation, the radius of rotation, and the angle of rotation
                xNew = curveCenterX + (np.cos(angRotMod) * -radius) 
                yNew = curveCenterY + (np.sin(angRotMod) * -radius)
            return np.array([xNew, yNew, thetaNew])

    def plot_position_DR(self, ax):
        # function to plot the robot path based on dead reckoning data
        xPosDR = []
        yPosDR = []
        for pos in self.posDeadRec:
            xPosDR.append(pos[0])
            yPosDR.append(pos[1])
        ax.plot(xPosDR, yPosDR, 'k-', label=' Robot Dead Reckoning Position')

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
        ax.scatter(xPos, yPos, c='g', label='Landmark Ground Truth Position')

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
        # Finds the heading from the robot to a given point (typically this will be a landmark)
        dist = math.dist([robX, robY], [pointX, pointY])
        phiGlobal = np.arctan2((pointY-robY),(pointX - robX))
        phiRob = phiGlobal - robTheta
        while phiRob < 0:
            phiRob += np.pi*2
        return dist, phiRob
    
    def reset_odometry_pos(self, xInit, yInit, thetaInit):
        # A function to clear stored odometry and position data
        self.odometry = np.array([0,0,0,0])
        self.posDeadRec = np.array([xInit, yInit, thetaInit, 0])
        self.posEstPF = np.array([xInit, yInit, thetaInit, 0])
        self.belState = np.array([])
        self.measurementPoints = np.array([0,0,0,0])

    def calculate_error_DR(self):
        # Calculates the dead reckoning error compared to the ground truth data
        print("Calculating dead reckoning error...")
        posIndex = 0
        # Iterate through the ground truth data and compute the errors at each step
        for measureGT in self.posGroundTruth:
            timeGT = measureGT[4]
            if timeGT < self.posDeadRec[0][3]:
                # If the groundtruth time is before the first calculated position time, do nothing
                pass
            else:
                # Otherwise, calculate the error based on the most recent calculated position
                posDR = self.posDeadRec[posIndex]
                # Use a while loop to find the next calculated position point
                while timeGT > posDR[3]:
                    posIndex += 1
                    posDR = self.posDeadRec[posIndex]
                # Calculate the distance between the ground truth and dead reckoning positions
                xGT = measureGT[0]
                yGT = measureGT[1]
                thetaGT = measureGT[2]
                xDR = posDR[0]
                yDR = posDR[1]
                thetaDR = posDR[2]
                errorDist = ((xDR - xGT)**2 + (yDR - yGT)**2)**0.5
                errorThetaRaw = thetaDR - thetaGT
                # Adjust the theta error to be between -pi and pi, then take the absolute value to find error magnitude
                while errorThetaRaw < -np.pi:
                    errorThetaRaw += 2*np.pi
                while errorThetaRaw > np.pi:
                    errorThetaRaw += -2*np.pi
                errorTheta = abs(errorThetaRaw)
                # Add error data to error array
                errorArray = np.array([errorDist, errorTheta, timeGT])
                self.errorDR = np.vstack([self.errorDR, errorArray])
        self.errorDR = np.delete(self.errorDR, 0, axis=0)
        print("Dead reckoning error calculation complete!")

    def plot_error_DR(self, ax, bx):
        # Plots the dead reckoning error
        time = []
        errorDist = []
        errorTheta = []
        # Iterate through the error data points and add them to the corresponding vectors
        for point in self.errorDR:
            time.append(point[2])
            errorTheta.append(point[1])
            errorDist.append(point[0])
        # Plot the data
        ax.plot(time, errorDist, 'b-', label='Dead Reckoning Position Error')
        bx.plot(time, errorTheta, 'b-', label='Dead Reckoning Angle Error')

    def calculate_error_PF(self):
        # Calculates the particle filter error compared to the ground truth data
        print("Calculating particle filter error...")
        posIndex = 0
        # Iterate through the ground truth data and compute the errors at each step
        for measureGT in self.posGroundTruth:
            timeGT = measureGT[4]
            if timeGT < self.posEstPF[0][3]:
                # If the groundtruth time is before the first calculated position time, do nothing
                pass
            else:
                # Otherwise, calculate the error based on the most recent calculated position
                posPF = self.posEstPF[posIndex]
                # Use a while loop to find the next calculated position point
                while timeGT > posPF[3]:
                    posIndex += 1
                    posPF = self.posEstPF[posIndex]
                # Calculate the distance between the ground truth and estimated positions
                xGT = measureGT[0]
                yGT = measureGT[1]
                thetaGT = measureGT[2]
                xPF = posPF[0]
                yPF = posPF[1]
                thetaPF = posPF[2]
                errorDist = ((xPF - xGT)**2 + (yPF - yGT)**2)**0.5
                errorThetaRaw = thetaPF - thetaGT
                # Adjust the theta error to be between -pi and pi, then take the absolute value to find error magnitude
                while errorThetaRaw < -np.pi:
                    errorThetaRaw += 2*np.pi
                while errorThetaRaw > np.pi:
                    errorThetaRaw += -2*np.pi
                errorTheta = abs(errorThetaRaw)
                # Add error data to error array
                errorArray = np.array([errorDist, errorTheta, timeGT])
                self.errorPF = np.vstack([self.errorPF, errorArray])
        self.errorPF = np.delete(self.errorPF, 0, axis=0)
        print("Particle filter error calculation complete!")

    def plot_error_PF(self, ax, bx):
        # Plots the particle filter error
        recentMeasureCutoff = 1
        time = []
        timeNoMeasure = []
        errorDist = []
        errorTheta = []
        errorDistNoMeasure = []
        errorThetaNoMeasure = []
        measureIndex = 1
        # Iterate through error data points and determine which vectors they should be added to
        for point in self.errorPF:
            errorTime = point[2]
            if errorTime > self.measurementPoints[0][3]:
                # If the time of the current error data point is within the range of the measurement points data,
                # this will determine if there was a measurement within [recentMeasureCutoff] seconds of the error point
                try:
                    timeMeasure = self.measurementPoints[measureIndex][3]
                    timeLastMeasure = self.measurementPoints[measureIndex-1][3]
                except IndexError as e:
                    timeLastMeasure = self.measurementPoints[measureIndex-1][3]
                endOfList = False
                while (point[2] > timeMeasure) and (endOfList == False):
                    try:
                        timeMeasure = self.measurementPoints[measureIndex][3]
                        timeLastMeasure = self.measurementPoints[measureIndex-1][3]
                        measureIndex += 1
                    except IndexError as e:
                        endOfList = True
                        timeLastMeasure = self.measurementPoints[measureIndex-1][3]
                timeStepLastMeasure = errorTime - timeLastMeasure
                # Check if there is a recent measurement and add the error point data to the corresponding vectors
                if timeStepLastMeasure > recentMeasureCutoff:
                    errorThetaNoMeasure.append(float(point[1]))
                    errorDistNoMeasure.append(float(point[0]))
                    timeNoMeasure.append(float(errorTime))
                    errorDist.append(point[0])
                    errorTheta.append(point[1])
                    time.append(errorTime)
                else:
                    errorDist.append(point[0])
                    errorTheta.append(point[1])
                    time.append(errorTime)
            else:
                # If the current error point is before the first measurement, add it to the "no recent measurement" vectors
                errorThetaNoMeasure.append(float(point[1]))
                errorDistNoMeasure.append(float(point[0]))
                timeNoMeasure.append(float(errorTime))
                errorTheta.append(point[1])
                errorDist.append(point[0])
                time.append(errorTime)
        # Plot the data
        print(errorThetaNoMeasure[-1])
        print(errorDistNoMeasure[-1])
        print(timeNoMeasure[-1])
        ax.scatter(timeNoMeasure, errorDistNoMeasure, c='r', label='Particle Filter Position Error, No Recent Measurement')
        bx.scatter(timeNoMeasure, errorThetaNoMeasure, c='r', label='Particle Filter Angle Error, No Recent Measurement')
        ax.plot(time, errorDist, 'g-', label='Particle Filter Position Error')
        bx.plot(time, errorTheta, 'g-', label='Particle Filter Angle Error')