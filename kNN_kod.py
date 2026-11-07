import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler


def normalizeData(df):
    # Extract columns to normalize (all except the last three)
    columnsToNormalize = df.columns[:-3]

    # Initialize scaler to transform values into the range [0, 1]
    scaler = MinMaxScaler()

    # Apply normalization only to selected columns
    df[columnsToNormalize] = scaler.fit_transform(df[columnsToNormalize])

    return df


def hyperparameterization(inputPath, outputPath, mapPath):
    # Load data
    df = pd.read_csv(inputPath)
    xCol, yCol = df.columns[-2], df.columns[-1]

    # Function to determine the zone based on coordinates
    def determineZone(x, y):
        if 0 <= x <= 156 and 0 <= y <= 88: return 1
        if 30 <= x <= 125 and 88 < y <= 198: return 2
        if 30 <= x <= 125 and 198 < y <= 333: return 3
        if 30 <= x <= 125 and 333 < y <= 512: return 4
        if 0 <= x <= 200 and 512 < y <= 655: return 5
        if 125 < x <= 162 and 88 <= y <= 512: return 6
        if 162 < x <= 417 and 41 <= y <= 128: return 7
        if 417 < x <= 708 and 41 <= y <= 128: return 8
        if 708 < x <= 1009 and 41 <= y <= 128: return 9
        if 1009 < x <= 1305 and 41 <= y <= 128: return 10
        if 1305 < x <= 1460 and 0 <= y <= 87: return 11
        if 1342 <= x <= 1437 and 87 < y <= 196: return 12
        if 1342 <= x <= 1437 and 196 < y <= 335: return 13
        if 1342 <= x <= 1437 and 335 < y <= 513: return 14
        if 1277 <= x <= 1460 and 513 < y <= 648: return 15
        if 1308 <= x <= 1345 and 86 <= y <= 511: return 16
        if 1169 <= x <= 1303 and 408 <= y <= 491: return 17
        if 1047 <= x <= 1169 and 408 <= y <= 491: return 18
        if 889 <= x <= 1047 and 408 <= y <= 491: return 19
        if 802 <= x <= 889 and 409 <= y <= 610: return 20
        if 553 <= x <= 670 and 409 <= y <= 610: return 21
        if 419 <= x <= 553 and 408 <= y <= 491: return 22
        if 294 <= x <= 419 and 409 <= y <= 491: return 23
        if 168 <= x <= 294 and 409 <= y <= 491: return 24
        if 163 <= x <= 1305 and 370 <= y <= 407: return 25
        if 305 <= x <= 459 and 164 <= y <= 367: return 26
        if 633 <= x <= 842 and 164 <= y <= 367: return 27
        if 1006 <= x <= 1172 and 164 <= y <= 367: return 28
        if 165 <= x <= 1305 and 129 <= y <= 163: return 29
        return 0

    # Add new zone column
    df['zone'] = [determineZone(x, y) for x, y in zip(df[xCol], df[yCol])]
    df.to_csv(outputPath, index=False)

    # Geometric data for drawing zone rectangles on the map
    zoneCoordinates = {
        1: [0, 0, 156, 88], 2: [30, 88, 95, 110], 3: [30, 198, 95, 135],
        4: [30, 333, 95, 179], 5: [0, 512, 200, 143], 6: [125, 88, 37, 424],
        7: [162, 41, 255, 87], 8: [417, 41, 291, 87], 9: [708, 41, 301, 87],
        10: [1009, 41, 296, 87], 11: [1305, 0, 155, 87], 12: [1342, 87, 95, 109],
        13: [1342, 196, 95, 139], 14: [1342, 335, 95, 178], 15: [1277, 513, 183, 135],
        16: [1308, 86, 37, 425], 17: [1169, 408, 134, 83], 18: [1047, 408, 122, 83],
        19: [889, 408, 158, 83], 20: [802, 409, 87, 201], 21: [553, 409, 117, 201],
        22: [419, 408, 134, 83], 23: [294, 409, 125, 82], 24: [168, 409, 126, 82],
        25: [163, 370, 1142, 37], 26: [305, 164, 154, 203], 27: [633, 164, 209, 203],
        28: [1006, 164, 166, 203], 29: [165, 129, 1140, 34]
    }

    fig, ax = plt.subplots(figsize=(15, 10))
    colorList = list(plt.cm.tab20b.colors + plt.cm.tab20c.colors)

    img = mpimg.imread(mapPath)
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.6, origin='upper')

    for zId, coord in zoneCoordinates.items():
        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3],
                                 linewidth=1.5, edgecolor='black',
                                 facecolor=colorList[zId], alpha=0.4)
        ax.add_patch(rect)
        ax.text(coord[0] + coord[2] / 2, coord[1] + coord[3] / 2, str(zId),
                color='black', fontweight='bold', ha='center', va='center',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.title("Zone Visualization")
    plt.tight_layout()
    plt.show()


def crossValidation(filePath, outputFolder='crossValidation'):
    # Split data into 5 independent folds to check model stability
    data = pd.read_csv(filePath)
    os.makedirs(outputFolder, exist_ok=True)

    totalSamples = len(data)
    testSize = totalSamples // 5
    shuffledData = data.sample(frac=1, random_state=42).reset_index(drop=True)

    for fold in range(5):
        setPath = os.path.join(outputFolder, f'set{fold + 1}')
        os.makedirs(setPath, exist_ok=True)

        testIndices = list(range(fold * testSize, (fold + 1) * testSize))
        trainIndices = [i for i in range(totalSamples) if i not in testIndices]

        shuffledData.iloc[testIndices].to_csv(os.path.join(setPath, f'testFold{fold + 1}.csv'), index=False)
        shuffledData.iloc[trainIndices].to_csv(os.path.join(setPath, f'trainFold{fold + 1}.csv'), index=False)


# Function to plot actual and predicted points on the map with error lines
def plotResultsOnMap(mapPath, yActual, yPredicted, setName):
    img = mpimg.imread(mapPath)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.7)

    ax.scatter(yActual[:, 0], yActual[:, 1], c='green', label='Actual', s=15, alpha=0.6)
    ax.scatter(yPredicted[:, 0], yPredicted[:, 1], c='red', label='Predicted', s=15, alpha=0.6)
    
    # Draw lines between predictions and actual points
    for i in range(len(yActual)):
        ax.plot([yActual[i, 0], yPredicted[i, 0]],
                [yActual[i, 1], yPredicted[i, 1]],
                color='red', linestyle='-', linewidth=0.7, alpha=0.4)

    plt.title(f"kNN Regression Results - {setName}")
    plt.legend()
    plt.show()


def runKNN(mainFolder, setName, trainPath, testPath, mapPath):
    folderSet = os.path.join(mainFolder, setName)
    folderGraphics = os.path.join(folderSet, "k_dependency_plots")
    os.makedirs(folderGraphics, exist_ok=True)

    dfTrain, dfTest = pd.read_csv(trainPath), pd.read_csv(testPath)
    rssiColumns = [i for i in dfTrain.columns if 'RSSI' in i]
    xCol, yCol, zoneCol = 'X Coordinate [0.1m]', 'Y Coordinate [0.1m]', 'zone'

    image = mpimg.imread(mapPath)
    imgHeight, imgWidth = image.shape[:2]

    def scaleCoord(p, maxV, pMin, pMax):
        return (p - pMin) / (pMax - pMin) * maxV

    xMin, xMax = dfTrain[xCol].min(), dfTrain[xCol].max()
    yMin, yMax = dfTrain[yCol].min(), dfTrain[yCol].max()

    trainImg = np.column_stack([scaleCoord(dfTrain[xCol].values, imgWidth, xMin, xMax),
                                scaleCoord(dfTrain[yCol].values, imgHeight, yMin, yMax)])
    testImg = np.column_stack([scaleCoord(dfTest[xCol].values, imgWidth, xMin, xMax),
                               scaleCoord(dfTest[yCol].values, imgHeight, yMin, yMax)])

    zoneModels = {}
    bestKPerZone = {}

    for z in np.unique(dfTrain[zoneCol]):
        maskTr, maskTe = dfTrain[zoneCol] == z, dfTest[zoneCol] == z
        if not np.any(maskTe) or not np.any(maskTr): continue

        xTr, yTr = dfTrain[rssiColumns].values[maskTr], trainImg[maskTr]
        xTe, yTe = dfTest[rssiColumns].values[maskTe], testImg[maskTe]

        kValues = range(1, min(len(xTr), 16))
        rmseForK = []

        for k in kValues:
            knn = KNeighborsRegressor(n_neighbors=k).fit(xTr, yTr)
            rmseForK.append(np.sqrt(mean_squared_error(yTe, knn.predict(xTe))))

        bestK = kValues[np.argmin(rmseForK)]
        bestKPerZone[z] = bestK
        zoneModels[z] = KNeighborsRegressor(n_neighbors=bestK).fit(xTr, yTr)

        plt.figure(figsize=(6, 4))
        plt.plot(kValues, rmseForK, 'b-o')
        plt.axvline(bestK, color='r', linestyle='--')
        plt.title(f"Zone {z} Best k={bestK}")
        plt.savefig(os.path.join(folderGraphics, f'zone_{z}_k.png'))
        plt.close()

    kRangeCls = range(1, min(21, len(dfTrain)))
    accuracyList = [accuracy_score(dfTest[zoneCol], KNeighborsClassifier(n_neighbors=k).fit(dfTrain[rssiColumns], dfTrain[zoneCol]).predict(dfTest[rssiColumns])) for k in kRangeCls]

    bestKCls = list(kRangeCls)[np.argmax(accuracyList)]

    modelCls = KNeighborsClassifier(n_neighbors=bestKCls).fit(dfTrain[rssiColumns], dfTrain[zoneCol])
    zonePred = modelCls.predict(dfTest[rssiColumns])
    
    individualErrors = []
    predictedCoordinates = []
    
    for i in range(len(testImg)):
        z = zonePred[i]
        prediction = zoneModels[z].predict([dfTest[rssiColumns].values[i]])[0] if z in zoneModels else \
            KNeighborsRegressor(n_neighbors=3).fit(dfTrain[rssiColumns], trainImg).predict(
                [dfTest[rssiColumns].values[i]])[0]
        predictedCoordinates.append(prediction)
        individualErrors.append(np.linalg.norm(prediction - testImg[i]))

    predictedCoordinates = np.array(predictedCoordinates)
    plotResultsOnMap(mapPath, testImg, predictedCoordinates, setName)

    errorArray = np.array(individualErrors)
    with open(os.path.join(folderSet, f"results_{setName}.txt"), "w") as f:
        f.write(
            f"STATISTICS FOR {setName}\nMean: {np.mean(errorArray):.2f}\nMedian: {np.median(errorArray):.2f}\nStd: {np.std(errorArray):.2f}\nVar: {np.var(errorArray):.2f}")

    return individualErrors


if __name__ == '__main__':
    dataset = pd.read_csv("original.csv")
    dataset.replace("*", -110, inplace=True)

    dataset.to_csv("originalII.csv", index=False)

    mapPath = r"Layout.gif"
    mainFolder = "Orig29"
    
    hyperparameterization('originalII.csv', 'hyperpar.csv', mapPath)
    crossValidation("hyperpar.csv", mainFolder)

    totalErrors = []
    for i in range(1, 6):
        setName = f'set{i}'

        trainPath = os.path.join(mainFolder, setName, f'trainFold{i}.csv')
        testPath = os.path.join(mainFolder, setName, f'testFold{i}.csv')

        setErrors = runKNN(mainFolder, setName, trainPath, testPath, mapPath)
        totalErrors.extend(setErrors)

    totalErrors = np.array(totalErrors)
    globalMean = np.mean(totalErrors)
    globalStd = np.std(totalErrors)

    print(f"RESULTS:")
    print(f"Mean: {globalMean:.2f}")
    print(f"Median: {np.median(totalErrors):.2f}")
    print(f"Std: {globalStd:.2f}")

    plt.hist(totalErrors, bins=40, color='skyblue', edgecolor='black')
    plt.title(f"Global Error Distribution (Mean: {globalMean:.2f}, Std: {globalStd:.2f})")
    plt.xlabel("Error [0.1m]")
    plt.ylabel("Sample Count")
    plt.savefig(os.path.join(mainFolder, 'global_statistics.png'))
    plt.show()
