import numpy as np
import matplotlib.pyplot as plt

class Map:
    def __init__(self):
        self.landmarkGT = {}
        pass

    def import_landmark_GT(self, filename): 
        # Imports data from specified file. File should be in same directory as the python files
        print('Importing landmark groundtruth data from file "' + filename + '"...')
        data = np.genfromtxt(filename)
        countMax = len(data)
        count = 0
        newData = {}
        while count + 1 < countMax:
            # Iterate through the data and calculate time steps based on the provided time data
            t0 = data[count][0]
            t1 = data[count + 1][0]
            tStep = t1 - t0
            newData[int(data[count][0])] = [data[count][1],data[count][2]]
            count += 1
        print("Landmark groundtruth data import complete!")
        self.landmarkGT = newData
        return newData

    def landmark_pos(self, barcode):
        # Returns the x,y coordinates for a landmark based on the provided barcode number
        [xPos, yPos] = self.landmarkGT[barcode]
        return [xPos, yPos]

    def plot_landmark_pos(self, ax):
        # Plots the locations on the landmarks
        xPos = []
        yPos = []
        for pos in self.landmarkGT:
            xPos.append(float(self.landmarkGT[pos][0]))
            yPos.append(float(self.landmarkGT[pos][1]))
        print(len(xPos), len(yPos))
        print([xPos,yPos])
        ax.scatter(xPos, yPos, c='k', label='Landmark Ground Truth Position')