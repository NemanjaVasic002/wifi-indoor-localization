# WiFi RSSI Indoor Localization

This project estimates the position of a device inside a building from the WiFi signal strength (RSSI) it receives from 8 access points. Every sample in the dataset has 8 RSSI values and the real x and y coordinates of the point where it was measured. The floor plan is divided into rectangular zones.

I tried three methods: a neural network, SVM and kNN. All three work in two steps. First a classifier predicts which zone the device is in, and then a regressor predicts the exact coordinates. The predicted zone is used to correct or narrow down the final position.

## Requirements

You need Python 3 and the following packages:

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

You can install them with:

```bash
pip install -r requirements.txt
```

TensorFlow is only used in the neural network script.

The scripts also need two files in the same folder:

`original.csv` is the dataset. The first 8 columns are the RSSI values, and their names have to contain the word RSSI because the kNN script finds them by name. The last two columns have to be named `X Coordinate [0.1m]` and `Y Coordinate [0.1m]`, and the coordinates are in units of 0.1 m. Missing RSSI values are marked with `*`, and every script replaces them with -110, which means there is no signal from that access point.

`Layout.gif` is the floor plan image, used as the background for the plots.

## How to run

```bash
python neuralNetworkLocalization.py
python svmLocalization.py
python knnLocalization.py
```

Each script runs on its own. It prepares the data, splits it into 5 folds for cross validation, trains the models, evaluates them and shows the plots. The plots open in separate windows, and the script continues only after you close them. The neural network trains for 1000 epochs per fold and the SVM script repeats cross validation 50 times, so both take a while to finish.

## neuralNetworkLocalization.py

This script uses two neural networks built in Keras. One classifies the zone (26 zones) and the other predicts the x and y coordinates. After prediction, each point is clipped to the bounds of the zone predicted by the classifier.

`rmseLoss` is a custom loss function that computes the RMSE between the real and predicted coordinates. It is used to train the regressor.

`plotResultsOnMap` draws the real points in green and the predicted points in red on top of the floor plan, and connects each pair with a line so the error is visible.

`Regressor` is the network for coordinate prediction. It has two hidden layers with 256 and 128 neurons, dropout of 0.2 and L2 regularization, and an output layer with 2 neurons for x and y. `_buildNetwork` creates and compiles it.

`MulticlassClassifier` is the network for zone classification. It has three hidden layers with 256, 128 and 64 neurons, and a softmax output with 26 classes. `train` trains it on a CSV file and keeps the epoch with the lowest training loss, which is saved to a .keras file. `evaluate` returns the accuracy in percent on the test file.

`ZONE_BOUNDS` holds the borders of each zone as xMin, xMax, yMin, yMax.

`runCrossValidation` goes through all 5 folds. For each fold it trains both networks, clips the predicted points to the predicted zone, draws the results and computes accuracy and RMSE. At the end it returns the average values.

`getZone` returns the zone number for a given x and y, or 0 if the point is not inside any zone. `assignZones` uses it to add a zone column to the whole dataset.

`splitIntoFolds` shuffles the data, splits it into 5 parts and saves the train and test file for each fold in the crossValidation folder.

The script creates originalNormalized.csv (data with `*` replaced), originalZoned.csv (data with the zone column), the crossValidation folder with the folds, and the saved models for each fold.

## svmLocalization.py

This script has the same structure as the previous one, but it uses SVC for classification and SVR for regression, both with an RBF kernel. The input data is standardized with StandardScaler. The folds are stratified, so every fold has the same share of each zone. The whole cross validation is repeated 50 times with a different seed each time, to check how stable the results are.

The settings are at the top of the file. `numExperiments` is the number of repetitions, `numFolds` is the number of folds and `baseSeed` is the starting seed. The classifier uses C = 100 and gamma = auto, and the regressor uses C = 700 and gamma = scale.

`rmseLoss` computes the RMSE with NumPy.

`getZoneFromBounds` returns the zone for one row of the table based on its coordinates, or 0 if the point is outside all zones.

`plotResultsOnMap` works the same way as in the neural network script.

`Regressor` combines StandardScaler with SVR, using MultiOutputRegressor so it can predict both x and y. It has `train` and `predict` methods.

`MulticlassClassifier` combines StandardScaler with SVC. It has `train`, `predict` and `evaluate`, and `evaluate` returns the accuracy in percent.

`runCrossValidation` trains and tests both models on every fold and clips the points to the predicted zone. It returns the accuracy and RMSE for each fold, together with all real and predicted zones. It draws the map only when asked to.

`splitIntoFolds` splits the data with StratifiedKFold and saves the folds.

At the end the script prints the mean, standard deviation and median of accuracy and RMSE, the accuracy for each zone separately, and two plots that show how accuracy and RMSE change between repetitions.

## knnLocalization.py

This script uses a more detailed division into 29 zones, which also includes the hallways. One kNN classifier predicts the zone, and each zone has its own kNN regressor with its own best value of k.

`normalize_data` scales all columns except the last three to the range from 0 to 1. It is defined but not used at the moment.

`hyperparameterization` assigns each sample to one of the 26 zones using the inner function `determine_zone`, and saves the data with the new zone column. It also draws all zones as colored rectangles on the floor plan, which makes it easy to check whether the zones are set correctly.

`cross_validation` shuffles the data with a fixed seed and splits it into 5 folds.

`plot_results_on_map` draws the real and predicted points with error lines.

`knn_localization` runs everything for one fold. First it converts the coordinates to pixels of the floor plan image with the inner function `scale_coord`. Then, for each zone, it tries k from 1 to 15, keeps the k with the lowest error and saves a plot of error versus k. After that it finds the best k from 1 to 20 for the zone classifier. For each test sample it predicts the zone and then the position, using the regressor of that zone. If the zone has no regressor of its own, it uses a general kNN with k = 3. The error statistics are saved to a text file and the function returns the error of every sample.

The script creates originalII.csv, hyperpar.csv and the Orig29 folder. Orig29 contains the folds, a results file and the k plots for each fold, and a histogram of all errors.

## Project structure

```
neuralNetworkLocalization.py
svmLocalization.py
knnLocalization.py
original.csv
Layout.gif
requirements.txt
README.md
```

## Results

Zones Segmentations

<img width="1142" height="548" alt="image" src="https://github.com/user-attachments/assets/2f595b28-30b7-4755-9e89-95e1b6865570" />

Zone accuracy and coordinate RMSE for each method:

SVM: zone accuracy 66.67% ± 2.19%, RMSE 84.79 ± 4.87 [0.1 m]

Neural network: zone accuracy 65.27%, RMSE 87.9 [0.1 m]

kNN: zone accuracy 64% to 72%, RMSE 92.4 [0.1 m]
