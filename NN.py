import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import load_model

# Column names as they appear in the input dataset
X_COLUMN = "X Coordinate [0.1m]"
Y_COLUMN = "Y Coordinate [0.1m]"
NUM_FEATURES = 8
NUM_ZONES = 26


# Computes the RMSE between true and predicted values
def rmseLoss(yTrue, yPred):
    return tf.sqrt(tf.reduce_mean(tf.square(yPred - yTrue)))


# Plots true and predicted points on the map, connected by error lines
def plotResultsOnMap(mapPath, yTrue, yPred, foldNumber):
    img = mpimg.imread(mapPath)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.7)

    ax.scatter(yTrue[:, 0], yTrue[:, 1], c="green", label="True", s=15, alpha=0.6)
    ax.scatter(yPred[:, 0], yPred[:, 1], c="red", label="Predicted", s=15, alpha=0.6)

    # Draw a line between each prediction and its true point
    for i in range(len(yTrue)):
        ax.plot([yTrue[i, 0], yPred[i, 0]],
                [yTrue[i, 1], yPred[i, 1]],
                color="red", linestyle="-", linewidth=0.7, alpha=0.4)

    plt.title(f"Regression Results with Error Lines - Fold {foldNumber}")
    plt.legend()
    plt.show()


# Neural network for coordinate regression
class Regressor:
    def __init__(self, numFeatures=NUM_FEATURES):
        self.numFeatures = numFeatures
        self.model = self._buildNetwork()

    def _buildNetwork(self):
        dropoutRate = 0.2
        l2Factor = 0.005
        model = models.Sequential([
            layers.Input(shape=(self.numFeatures,)),
            layers.Dense(256, kernel_regularizer=regularizers.l2(l2Factor), activation="relu"),
            layers.Dropout(dropoutRate),
            layers.Dense(128, kernel_regularizer=regularizers.l2(l2Factor), activation="relu"),
            layers.Dropout(dropoutRate),
            layers.Dense(2, activation="linear")
        ])
        model.compile(optimizer="adam", loss=rmseLoss, metrics=["mae"])
        return model


# Neural network for zone classification
class MulticlassClassifier:
    def __init__(self, numClasses=NUM_ZONES, numFeatures=NUM_FEATURES):
        self.numClasses = numClasses
        self.numFeatures = numFeatures
        self.model = self._buildNetwork()

    def _buildNetwork(self):
        dropoutRate = 0.2
        l2Factor = 0.005
        model = models.Sequential([
            layers.Input(shape=(self.numFeatures,)),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2Factor)),
            layers.Dropout(dropoutRate),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2Factor)),
            layers.Dropout(dropoutRate),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2Factor)),
            layers.Dropout(dropoutRate),
            layers.Dense(self.numClasses, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    # Trains the classifier on the given CSV file
    def train(self, csvPath, epochs=1000, foldIndex=0):
        df = pd.read_csv(csvPath)
        X = df.iloc[:, :self.numFeatures].values
        y = df["zone"].values - 1
        checkpointPath = f"bestMulticlassFold{foldIndex}.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpointPath, monitor="loss", save_best_only=True, verbose=0)
        self.model.fit(X, y, epochs=epochs, verbose=0, batch_size=64, callbacks=[checkpoint])
        self.model = load_model(checkpointPath)

    # Evaluates classifier accuracy (%) on the test CSV file
    def evaluate(self, testCsvPath):
        dfTest = pd.read_csv(testCsvPath)
        xTest = dfTest.iloc[:, :self.numFeatures].values
        yTrue = dfTest["zone"].values - 1
        predictions = self.model.predict(xTest, verbose=0)
        predictedZones = np.argmax(predictions, axis=1)
        correctHits = np.sum(predictedZones == yTrue)
        return (correctHits / len(yTrue)) * 100


# Boundaries of each zone on the map [xMin, xMax, yMin, yMax]
ZONE_BOUNDS = {
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


# Runs cross-validation over the given number of folds
def runCrossValidation(mainPath, mapPath, numFolds=5):
    accuracyList = []
    rmseList = []

    for i in range(1, numFolds + 1):
        print(f"\nFold {i}")
        foldPath = os.path.join(mainPath, f"fold{i}")
        trainPath = os.path.join(foldPath, f"trainFold{i}.csv")
        testPath = os.path.join(foldPath, f"testFold{i}.csv")

        dfTrain = pd.read_csv(trainPath)
        xTrain = dfTrain.iloc[:, :NUM_FEATURES].values
        yTrainReg = dfTrain[[X_COLUMN, Y_COLUMN]].values

        dfTest = pd.read_csv(testPath)
        xTest = dfTest.iloc[:, :NUM_FEATURES].values
        yTrue = dfTest[[X_COLUMN, Y_COLUMN]].values

        # Zone classification
        classifier = MulticlassClassifier(numClasses=NUM_ZONES, numFeatures=NUM_FEATURES)
        classifier.train(trainPath, epochs=1000, foldIndex=i)
        accuracy = classifier.evaluate(testPath)
        accuracyList.append(accuracy)

        # Coordinate regression
        regressor = Regressor(numFeatures=NUM_FEATURES)
        regCheckpointPath = f"bestRegressorFold{i}.keras"
        regCheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=regCheckpointPath, monitor="loss", save_best_only=True, verbose=0)
        regressor.model.fit(xTrain, yTrainReg, epochs=1000, verbose=0, batch_size=64, callbacks=[regCheckpoint])
        regressor.model = load_model(regCheckpointPath, custom_objects={"rmseLoss": rmseLoss})

        yPredRaw = regressor.model.predict(xTest, verbose=0)
        classPredictions = classifier.model.predict(xTest, verbose=0)
        predictedZones = np.argmax(classPredictions, axis=1) + 1

        # Clip each regressed point to the bounds of its predicted zone
        yFinal = []
        for idx, point in enumerate(yPredRaw):
            zoneId = predictedZones[idx]
            if zoneId in ZONE_BOUNDS:
                xMin, xMax, yMin, yMax = ZONE_BOUNDS[zoneId]
                xClipped = np.clip(point[0], xMin, xMax)
                yClipped = np.clip(point[1], yMin, yMax)
                yFinal.append([xClipped, yClipped])
            else:
                yFinal.append(point)

        yFinal = np.array(yFinal)

        plotResultsOnMap(mapPath, yTrue, yFinal, i)

        foldRmse = np.sqrt(np.mean(np.square(yFinal - yTrue)))
        rmseList.append(foldRmse)
        print(f"Fold {i} Accuracy: {accuracy:.2f}%, RMSE: {foldRmse:.2f}")

    return np.mean(accuracyList), np.mean(rmseList)


# Determines the zone ID for a point (0 if it is outside all zones)
def getZone(x, y):
    if 0 <= x <= 162 and 0 <= y <= 88: return 1
    if 30 <= x <= 162 and 88 < y <= 198: return 2
    if 30 <= x <= 162 and 198 < y <= 333: return 3
    if 30 <= x <= 162 and 333 < y <= 512: return 4
    if 0 <= x <= 200 and 512 < y <= 655: return 5
    if 162 < x <= 417 and 41 <= y <= 163: return 6
    if 417 < x <= 708 and 41 <= y <= 163: return 7
    if 708 < x <= 1009 and 41 <= y <= 163: return 8
    if 1009 < x <= 1308 and 41 <= y <= 163: return 9
    if 1305 < x <= 1460 and 0 <= y <= 87: return 10
    if 1308 <= x <= 1437 and 87 < y <= 196: return 11
    if 1308 <= x <= 1437 and 196 < y <= 335: return 12
    if 1308 <= x <= 1437 and 335 < y <= 513: return 13
    if 1277 <= x <= 1460 and 513 < y <= 648: return 14
    if 1169 <= x <= 1308 and 370 <= y <= 491: return 15
    if 1047 <= x <= 1169 and 370 <= y <= 491: return 16
    if 889 <= x <= 1047 and 370 <= y <= 491: return 17
    if 802 <= x <= 889 and 370 <= y <= 610: return 18
    if 553 <= x <= 670 and 370 <= y <= 610: return 19
    if 419 <= x <= 553 and 370 <= y <= 491: return 20
    if 294 <= x <= 419 and 370 <= y <= 491: return 21
    if 162 <= x <= 294 and 370 <= y <= 491: return 22
    if 305 <= x <= 459 and 164 <= y <= 367: return 23
    if 633 <= x <= 842 and 164 <= y <= 367: return 24
    if 1006 <= x <= 1172 and 164 <= y <= 367: return 25
    if 670 < x < 802 and 370 < y <= 412: return 26
    return 0


# Adds a zone column to the dataset based on the coordinates
def assignZones(inputPath, outputPath):
    df = pd.read_csv(inputPath)
    xCol, yCol = df.columns[-2], df.columns[-1]
    df["zone"] = [getZone(x, y) for x, y in zip(df[xCol], df[yCol])]
    df.to_csv(outputPath, index=False)


# Splits the data into 5 folds for cross-validation
def splitIntoFolds(filePath, outputFolder="crossValidation", numFolds=5):
    data = pd.read_csv(filePath)
    os.makedirs(outputFolder, exist_ok=True)
    totalSamples = len(data)
    testSize = totalSamples // numFolds
    shuffledData = data.sample(frac=1).reset_index(drop=True)

    for fold in range(numFolds):
        foldPath = os.path.join(outputFolder, f"fold{fold + 1}")
        os.makedirs(foldPath, exist_ok=True)
        testIndices = set(range(fold * testSize, (fold + 1) * testSize))
        trainIndices = [i for i in range(totalSamples) if i not in testIndices]
        shuffledData.iloc[sorted(testIndices)].to_csv(
            os.path.join(foldPath, f"testFold{fold + 1}.csv"), index=False)
        shuffledData.iloc[trainIndices].to_csv(
            os.path.join(foldPath, f"trainFold{fold + 1}.csv"), index=False)


if __name__ == "__main__":
    # Path to the floor plan image
    mapPath = "Layout.gif"
    # Input data
    inputPath = "original.csv"
    # Output after replacing unknown values with -110
    normalizedPath = "originalNormalized.csv"
    # Output after assigning zones
    zonedPath = "originalZoned.csv"
    # Folder where the 5-fold cross-validation splits are stored
    crossValidationPath = "crossValidation"

    # Replace unknown values ("*") with -110
    dataSet = pd.read_csv(inputPath)
    dataSet.replace("*", -110, inplace=True)
    dataSet.to_csv(normalizedPath, index=False)

    # Assign a zone to every sample
    assignZones(normalizedPath, zonedPath)

    globalAccuracy = []
    globalRmse = []
    numExperiments = 10

    for k in range(1, numExperiments + 1):
        print(f"Experiment {k}/{numExperiments}")
        # Split into 5 folds
        splitIntoFolds(zonedPath, crossValidationPath)
        # Get mean metrics across folds
        meanAccuracy, meanRmse = runCrossValidation(crossValidationPath, mapPath, numFolds=5)
        globalAccuracy.append(meanAccuracy)
        globalRmse.append(meanRmse)

    print(f"Global mean accuracy: {np.mean(globalAccuracy):.2f}%")
    print(f"Global accuracy STD: {np.std(globalAccuracy):.2f}%")
    print(f"Global median accuracy: {np.median(globalAccuracy):.2f}%")

    print(f"Global mean RMSE: {np.mean(globalRmse):.2f}")
    print(f"Global RMSE STD: {np.std(globalRmse):.2f}")
    print(f"Global median RMSE: {np.median(globalRmse):.2f}")
