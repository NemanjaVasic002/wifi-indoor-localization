# wifi-indoor-localization
Comparative analysis of ML models (KNN, Neural Networks, SVM) for WiFi RSSI indoor localization, including missing data imputation and spatial terrain segmentation.
Methodology
Data Preprocessing: Replaces unknown or missing hardware readings (*) with the hardware sensitivity threshold (-110 dBm).
Zone Hyperparameterization: Automatically maps spatial (X, Y) coordinates into predefined structural zones to partition the search space.
Hybrid Modeling:Classification: Predicts the target zone using a K-Nearest Neighbors Classifier.
                Regression: Predicts precise coordinates within the predicted zone using an optimized K-Nearest Neighbors Regressor.
Cross-Validation: Evaluates robustness using a 5-fold cross-validation scheme.Visualization: Generates spatial error distribution plots and maps actual vs. predicted coordinates with connecting error lines.




Requirements

Install the required Python packages:
Bash

pip install numpy pandas scikit-learn matplotlib
