import os, sys, pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def weightGauss(value, sigma, mue=0.0):
    weight = np.exp(-(np.square(value - mue) / (2 * np.square(sigma))))

    return weight

def FilterMergedVelocities(veloTable, NN_nbr, searchRadius):
    neighborsTree = NearestNeighbors(n_neighbors=NN_nbr, algorithm='kd_tree').fit(veloTable.loc[:,['X','Y','Z']])
    distances, indices = neighborsTree.kneighbors(veloTable.loc[:,['X','Y','Z']])

    for i in range(indices.shape[0]):
        #keep only values within search radius
        velos = np.asarray(veloTable.loc[indices[i, :], ['velo']]).flatten()
        distancesVelo = distances[i]
        veloDist = np.vstack((velos, distancesVelo)).T
        veloDist = veloDist[veloDist[:,1] < searchRadius]
        velos = veloDist[:,0]
        veloDistances = veloDist[:,1]

        #get std velocities
        stdVelo = np.std(veloDist[:,0])
        medianVelo = np.median(veloDist[:,0])

        # get weights for velocity based on velocity distribution
        weightsVeloGauss = []
        for velocity in velos:
            weightsVeloGauss.append(weightGauss(velocity, sigma=stdVelo, mue=medianVelo))
        weightsVeloGauss = np.asarray(weightsVeloGauss)
        weightsVeloGauss = weightsVeloGauss / (np.ones((weightsVeloGauss.shape[0])) * np.sum(weightsVeloGauss))

        #get weights for velocity based on distance
        weightsDistGauss = []
        for distanceVelo in veloDistances:
            weightsDistGauss.append(weightGauss(distanceVelo, sigma=0.2))
        weightsDistGauss = np.asarray(weightsDistGauss)
        weightsDistGauss = weightsDistGauss / (np.ones((weightsDistGauss.shape[0])) * np.sum(weightsDistGauss))

        #combine distance and velocity based weights to filter velocity
        velos = veloDist[:,0]
        filteredVelo = np.sum(velos * ((1/3 * weightsDistGauss + 2/3 * weightsVeloGauss)))
        veloTable.loc[i,['velo']] = filteredVelo

    return veloTable

def AllTracksOneFile(directory, fileName):
    if os.path.isdir(directory):
        for dirpath, dirsubpaths, dirfiles in os.walk(directory):
            if len(dirsubpaths) >= 1:
                break
            else:
                print('empty directory: ' + dirpath)
                sys.exit()
    else:
        print('directory ' + directory + ' not found')
        sys.exit()

    firstLoop = True
    for dir in dirsubpaths:
        if 'velocities' in dir:
            try:
            # for file in os.listdir(dirpath + dir + '/'):
            #     if "tracksFiltered_locally_PTV.txt" == file:
                if firstLoop:
                    frame = pd.read_csv(dirpath + dir + '/' + fileName, delimiter='\t')
                    firstLoop = False
                    continue
                frame_new = pd.read_csv(dirpath + dir + '/' + fileName, delimiter='\t')
                frame = pd.concat([frame, frame_new])
            except Exception as e:
                print(e)
                continue

    return frame

dirFiles = ".../YourOutputFolderChoice/"
fileName = "tracksFiltered_locally_PTV.txt"
NN_nbr = 30
searchRadius = 3

#merge all velocity files
allTracks = AllTracksOneFile(dirFiles, fileName)
allTracks = allTracks.reset_index()
allTracks = allTracks.drop(columns=['index'])
allTracks.to_csv(dirFiles + "TracksFilteredVeloRaw.txt")

#filter velocity based on standard deviation of velocity and distance to source velocity
allTracksFiltered = FilterMergedVelocities(allTracks, NN_nbr=20, searchRadius=2)
allTracksFiltered.to_csv(dirFiles + "TracksFilteredVeloGauss_" + str(NN_nbr) + "_" + str(searchRadius) + ".txt")