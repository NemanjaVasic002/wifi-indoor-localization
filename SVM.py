import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Paths
mapPath = "Layout.gif"
inputFile = "original.csv"
normalizedFile = "originalNormalized.csv"
zonedFile = "originalZoned.csv"
crossValidationPath = "crossValidation"

# Experiment settings
numExperiments = 50
numFolds = 5
baseSeed = 42

# Classifier hyperparameters
clfKernel = "rbf"
clfC = 100
clfGamma = "auto"

# Regressor hyperparameters
regKernel = "rbf"
regC = 700
regGamma = "scale"

# Column names as they appear in the input dataset
xColumn = "X Coordinate [0.1m]"
yColumn = "Y Coordinate [0.1m]"
numFeatures = 8

# Boundaries of each zone on the map [xMin, xMax, yMin, yMax]
zoneBounds = {
    1: [0, 162, 0, 88], 2: [30, 162, 88, 198], 3: [30, 162, 198, 333],
    4: [30, 162, 333, 512], 5: [0, 200, 512, 655], 6: [162, 417, 41, 163],
    7: [417, 708, 41, 163], 8: [708, 1009, 41, 163], 9: [1009, 1308, 41, 163],
    10: [1305, 1460, 0, 87], 11: [1308, 1437, 87, 196], 12: [1308, 1437, 196, 335],
    13: [1308, 1437, 335, 513], 14: [1277, 1460, 513, 648], 15: [1169, 1308, 370, 491],
    16: [1047, 1169, 370, 491], 17: [889, 1047, 370, 491], 18: [802, 889, 370, 610],
    19: [553, 670, 370, 610], 20: [419, 553, 370, 491], 21: [294, 419, 370, 491],
    22: [162, 294, 370, 491], 23: [305, 459, 164, 367], 24: [633, 842, 164, 367],
    25: [1006, 1172, 164, 367], 26: [670, 802, 370, 412]
}


# Computes the RMSE between true and predicted values
def rmseLoss(yTrue, yPred):
    return np.sqrt(np.mean(np.square(yPred - yTrue)))


# Returns the zone ID for a row based on its coordinates (0 if outside all zones)
def getZoneFromBounds(row):
    x = row[xColumn]
    y = row[yColumn]
    for zoneId, bounds in zoneBounds.items():
        if bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]:
            return zoneId
    return 0


# Plots true and predicted points on the map, connected by error lines
def plotResultsOnMap(mapFile, yTrue, yPred, foldNumber):
    img = mpimg.imread(mapFile)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.7)

    ax.scatter(yTrue[:, 0], yTrue[:, 1], c="green", label="True", s=15, alpha=0.6)
    ax.scatter(yPred[:, 0], yPred[:, 1], c="red", label="Predicted", s=15, alpha=0.6)

    for i in range(len(yTrue)):
        ax.plot(
            [yTrue[i, 0], yPred[i, 0]],
            [yTrue[i, 1], yPred[i, 1]],
            color="red", linestyle="-", linewidth=0.7, alpha=0.4
        )

    plt.title(f"Results - Fold {foldNumber}")
    plt.legend()
    plt.show()


# SVR-based regressor for (x, y) coordinates
class Regressor:
    def __init__(self):
        self.model = MultiOutputRegressor(
            SVR(kernel=regKernel, C=regC, gamma=regGamma)
        )
        self.scaler = StandardScaler()

    def train(self, X, y):
        xScaled = self.scaler.fit_transform(X)
        self.model.fit(xScaled, y)

    def predict(self, X):
        xScaled = self.scaler.transform(X)
        return self.model.predict(xScaled)


# SVC-based multiclass zone classifier
class MulticlassClassifier:
    def __init__(self):
        self.model = SVC(kernel=clfKernel, C=clfC, gamma=clfGamma)
        self.scaler = StandardScaler()

    def train(self, X, y):
        xScaled = self.scaler.fit_transform(X)
        self.model.fit(xScaled, y)

    def predict(self, X):
        xScaled = self.scaler.transform(X)
        return self.model.predict(xScaled)

    # Returns classification accuracy in percent
    def evaluate(self, xTest, yTrue):
        pred = self.predict(xTest)
        correct = np.sum(pred == yTrue)
        return (correct / len(yTrue)) * 100


# Runs cross-validation over all folds and returns per-fold metrics
def runCrossValidation(mainPath, mapFile, numFolds=5, drawMap=False):
    accuracyList = []
    rmseList = []
    trueClasses = []
    predictedClasses = []

    for i in range(1, numFolds + 1):
        print(f"\nFold {i}")

        foldPath = os.path.join(mainPath, f"fold{i}")
        trainPath = os.path.join(foldPath, f"trainFold{i}.csv")
        testPath = os.path.join(foldPath, f"testFold{i}.csv")

        dfTrain = pd.read_csv(trainPath)
        xTrain = dfTrain.iloc[:, :numFeatures].values
        yTrainClf = dfTrain["zone"].values
        yTrainReg = dfTrain[[xColumn, yColumn]].values

        dfTest = pd.read_csv(testPath)
        xTest = dfTest.iloc[:, :numFeatures].values
        yTestClf = dfTest["zone"].values
        yTrue = dfTest[[xColumn, yColumn]].values

        # Zone classification
        classifier = MulticlassClassifier()
        classifier.train(xTrain, yTrainClf)
        accuracy = classifier.evaluate(xTest, yTestClf)
        accuracyList.append(accuracy)

        # Coordinate regression
        regressor = Regressor()
        regressor.train(xTrain, yTrainReg)
        yPredRaw = regressor.predict(xTest)

        predictedZones = classifier.predict(xTest)
        trueClasses.extend(yTestClf)
        predictedClasses.extend(predictedZones)

        # Clip each regressed point to the bounds of its predicted zone
        yFinal = []
        for idx, point in enumerate(yPredRaw):
            zoneId = predictedZones[idx]
            xP, yP = point[0], point[1]

            if zoneId in zoneBounds:
                xMin, xMax, yMin, yMax = zoneBounds[zoneId]
                xClipped = max(xMin, min(xP, xMax))
                yClipped = max(yMin, min(yP, yMax))
                yFinal.append([xClipped, yClipped])
            else:
                yFinal.append([xP, yP])

        yFinal = np.array(yFinal)

        rmse = rmseLoss(yTrue, yFinal)
        rmseList.append(rmse)

        print(f"Fold {i} Accuracy: {accuracy:.2f}% | RMSE: {rmse:.2f}")

        if drawMap and i == numFolds:
            plotResultsOnMap(mapFile, yTrue, yFinal, i)

    return accuracyList, rmseList, trueClasses, predictedClasses


# Splits the data into stratified folds (by zone) and saves them to disk
def splitIntoFolds(filePath, outputFolder="crossValidation", numFolds=5, seed=42):
    data = pd.read_csv(filePath)
    os.makedirs(outputFolder, exist_ok=True)

    yClf = data["zone"].values
    skf = StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=seed)

    for fold, (trainIdx, testIdx) in enumerate(skf.split(data, yClf)):
        foldPath = os.path.join(outputFolder, f"fold{fold + 1}")
        os.makedirs(foldPath, exist_ok=True)

        data.iloc[trainIdx].to_csv(os.path.join(foldPath, f"trainFold{fold + 1}.csv"), index=False)
        data.iloc[testIdx].to_csv(os.path.join(foldPath, f"testFold{fold + 1}.csv"), index=False)


if __name__ == "__main__":
    # Replace unknown values ("*") with -110
    dataSet = pd.read_csv(inputFile)
    dataSet.replace("*", -110, inplace=True)
    dataSet.to_csv(normalizedFile, index=False)

    # Assign a zone to every sample
    zonedDataSet = dataSet.copy()
    zonedDataSet["zone"] = zonedDataSet.apply(getZoneFromBounds, axis=1)
    zonedDataSet.to_csv(zonedFile, index=False)

    globalAccuracy = []
    globalRmse = []
    meanAccuracyPerExperiment = []
    meanRmsePerExperiment = []
    globalTrueClasses = []
    globalPredictedClasses = []

    for k in range(1, numExperiments + 1):
        print(f"Experiment {k}/{numExperiments}")

        splitIntoFolds(zonedFile, crossValidationPath, numFolds=numFolds, seed=baseSeed + k)

        isLast = (k == numExperiments)
        accuracyList, rmseList, trueClasses, predictedClasses = runCrossValidation(
            crossValidationPath, mapPath, numFolds=numFolds, drawMap=isLast
        )

        globalAccuracy.extend(accuracyList)
        globalRmse.extend(rmseList)
        meanAccuracyPerExperiment.append(np.mean(accuracyList))
        meanRmsePerExperiment.append(np.mean(rmseList))
        globalTrueClasses.extend(trueClasses)
        globalPredictedClasses.extend(predictedClasses)

    print(f"\nAccuracy mean: {np.mean(globalAccuracy):.2f}%")
    print(f"Accuracy std: {np.std(globalAccuracy):.2f}%")
    print(f"Accuracy median: {np.median(globalAccuracy):.2f}%")

    print(f"\nRMSE mean: {np.mean(globalRmse):.2f}")
    print(f"RMSE std: {np.std(globalRmse):.2f}")
    print(f"RMSE median: {np.median(globalRmse):.2f}")

    # Per-zone accuracy across all experiments
    print("\nAccuracy per zone:")
    trueArray = np.array(globalTrueClasses)
    predictedArray = np.array(globalPredictedClasses)
    uniqueClasses = np.unique(trueArray)

    for classId in sorted(uniqueClasses):
        classMask = (trueArray == classId)
        correctPredictions = np.sum(predictedArray[classMask] == classId)
        totalForClass = np.sum(classMask)
        classAccuracy = (correctPredictions / totalForClass) * 100
        print(f"Zone {classId}: {classAccuracy:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, numExperiments + 1), meanAccuracyPerExperiment, color="blue")
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy (%)")
    plt.title("Mean Accuracy per Experiment")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, numExperiments + 1), meanRmsePerExperiment, color="orange")
    plt.xlabel("Experiment")
    plt.ylabel("RMSE [0.1m]")
    plt.title("Mean RMSE per Experiment")
    plt.show()