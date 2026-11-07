import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler


def normalize_data(df):
    # Extract columns to normalize (all except the last three)
    columns_to_normalize = df.columns[:-3]

    # Initialize scaler to transform values into the range [0, 1]
    scaler = MinMaxScaler()

    # Apply normalization only to selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df


def hyperparameterization(input_path, output_path, map_path):
    # Load data
    df = pd.read_csv(input_path)
    x_col, y_col = df.columns[-2], df.columns[-1]

    # Function to determine the zone based on coordinates
    def determine_zone(x, y):
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
    df['zone'] = [determine_zone(x, y) for x, y in zip(df[x_col], df[y_col])]
    df.to_csv(output_path, index=False)

    # Geometric data for drawing zone rectangles on the map
    zone_coordinates = {
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
    color_list = list(plt.cm.tab20b.colors + plt.cm.tab20c.colors)

    img = mpimg.imread(map_path)
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.6, origin='upper')

    for z_id, coord in zone_coordinates.items():
        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3],
                                 linewidth=1.5, edgecolor='black',
                                 facecolor=color_list[z_id], alpha=0.4)
        ax.add_patch(rect)
        ax.text(coord[0] + coord[2] / 2, coord[1] + coord[3] / 2, str(z_id),
                color='black', fontweight='bold', ha='center', va='center',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.title("Zone Visualization")
    plt.tight_layout()
    plt.show()


def cross_validation(file_path, output_folder='crossValidation'):
    # Split data into 5 independent folds to check model stability
    data = pd.read_csv(file_path)
    os.makedirs(output_folder, exist_ok=True)

    total_samples = len(data)
    test_size = total_samples // 5
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    for fold in range(5):
        set_path = os.path.join(output_folder, f'set{fold + 1}')
        os.makedirs(set_path, exist_ok=True)

        test_indices = list(range(fold * test_size, (fold + 1) * test_size))
        train_indices = [i for i in range(total_samples) if i not in test_indices]

        shuffled_data.iloc[test_indices].to_csv(os.path.join(set_path, f'testFold{fold + 1}.csv'), index=False)
        shuffled_data.iloc[train_indices].to_csv(os.path.join(set_path, f'trainFold{fold + 1}.csv'), index=False)


# Function to plot actual and predicted points on the map with error lines
def plot_results_on_map(map_path, y_actual, y_predicted, set_name):
    img = mpimg.imread(map_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=0.7)

    ax.scatter(y_actual[:, 0], y_actual[:, 1], c='green', label='Actual', s=15, alpha=0.6)
    ax.scatter(y_predicted[:, 0], y_predicted[:, 1], c='red', label='Predicted', s=15, alpha=0.6)

    # Draw lines between predictions and actual points
    for i in range(len(y_actual)):
        ax.plot([y_actual[i, 0], y_predicted[i, 0]],
                [y_actual[i, 1], y_predicted[i, 1]],
                color='red', linestyle='-', linewidth=0.7, alpha=0.4)

    plt.title(f"kNN Regression Results - {set_name}")
    plt.legend()
    plt.show()


def knn_localization(main_folder, set_name, train_path, test_path, map_path):
    folder_set = os.path.join(main_folder, set_name)
    folder_graphics = os.path.join(folder_set, "k_dependency_plots")
    os.makedirs(folder_graphics, exist_ok=True)

    df_train, df_test = pd.read_csv(train_path), pd.read_csv(test_path)
    rssi_columns = [i for i in df_train.columns if 'RSSI' in i]
    x_col, y_col, zone_col = 'X Coordinate [0.1m]', 'Y Coordinate [0.1m]', 'zone'

    image = mpimg.imread(map_path)
    img_height, img_width = image.shape[:2]

    def scale_coord(p, max_v, p_min, p_max):
        return (p - p_min) / (p_max - p_min) * max_v

    x_min, x_max = df_train[x_col].min(), df_train[x_col].max()
    y_min, y_max = df_train[y_col].min(), df_train[y_col].max()

    train_img = np.column_stack([scale_coord(df_train[x_col].values, img_width, x_min, x_max),
                                 scale_coord(df_train[y_col].values, img_height, y_min, y_max)])
    test_img = np.column_stack([scale_coord(df_test[x_col].values, img_width, x_min, x_max),
                                scale_coord(df_test[y_col].values, img_height, y_min, y_max)])

    zone_models = {}
    best_k_per_zone = {}

    for z in np.unique(df_train[zone_col]):
        mask_tr, mask_te = df_train[zone_col] == z, df_test[zone_col] == z
        if not np.any(mask_te) or not np.any(mask_tr): continue

        x_tr, y_tr = df_train[rssi_columns].values[mask_tr], train_img[mask_tr]
        x_te, y_te = df_test[rssi_columns].values[mask_te], test_img[mask_te]

        k_values = range(1, min(len(x_tr), 16))
        rmse_for_k = []

        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k).fit(x_tr, y_tr)
            rmse_for_k.append(np.sqrt(mean_squared_error(y_te, knn.predict(x_te))))

        best_k = k_values[np.argmin(rmse_for_k)]
        best_k_per_zone[z] = best_k
        zone_models[z] = KNeighborsRegressor(n_neighbors=best_k).fit(x_tr, y_tr)

        plt.figure(figsize=(6, 4))
        plt.plot(k_values, rmse_for_k, 'b-o')
        plt.axvline(best_k, color='r', linestyle='--')
        plt.title(f"Zone {z} Best k={best_k}")
        plt.savefig(os.path.join(folder_graphics, f'zone_{z}_k.png'))
        plt.close()

    k_range_cls = range(1, min(21, len(df_train)))
    accuracy_list = [accuracy_score(df_test[zone_col], KNeighborsClassifier(n_neighbors=k).fit(df_train[rssi_columns],
                                                                                               df_train[
                                                                                                   zone_col]).predict(
        df_test[rssi_columns])) for k in k_range_cls]

    best_k_cls = list(k_range_cls)[np.argmax(accuracy_list)]

    model_cls = KNeighborsClassifier(n_neighbors=best_k_cls).fit(df_train[rssi_columns], df_train[zone_col])
    zone_pred = model_cls.predict(df_test[rssi_columns])

    individual_errors = []
    predicted_coordinates = []

    for i in range(len(test_img)):
        z = zone_pred[i]
        prediction = zone_models[z].predict([df_test[rssi_columns].values[i]])[0] if z in zone_models else \
            KNeighborsRegressor(n_neighbors=3).fit(df_train[rssi_columns], train_img).predict(
                [df_test[rssi_columns].values[i]])[0]
        predicted_coordinates.append(prediction)
        individual_errors.append(np.linalg.norm(prediction - test_img[i]))

    predicted_coordinates = np.array(predicted_coordinates)
    plot_results_on_map(map_path, test_img, predicted_coordinates, set_name)

    error_array = np.array(individual_errors)
    with open(os.path.join(folder_set, f"results_{set_name}.txt"), "w") as f:
        f.write(
            f"STATISTICS FOR {set_name}\nMean: {np.mean(error_array):.2f}\nMedian: {np.median(error_array):.2f}\nStd: {np.std(error_array):.2f}\nVar: {np.var(error_array):.2f}")

    return individual_errors


if __name__ == '__main__':
    dataset = pd.read_csv("original.csv")
    dataset.replace("*", -110, inplace=True)

    dataset.to_csv("originalII.csv", index=False)

    map_path = r"Layout.gif"
    main_folder = "Orig29"

    hyperparameterization('originalII.csv', 'hyperpar.csv', map_path)
    cross_validation("hyperpar.csv", main_folder)

    total_errors = []
    for i in range(1, 6):
        set_name = f'set{i}'

        train_path = os.path.join(main_folder, set_name, f'trainFold{i}.csv')
        test_path = os.path.join(main_folder, set_name, f'testFold{i}.csv')

        set_errors = knn_localization(main_folder, set_name, train_path, test_path, map_path)
        total_errors.extend(set_errors)

    total_errors = np.array(total_errors)
    global_mean = np.mean(total_errors)
    global_std = np.std(total_errors)

    print(f"RESULTS:")
    print(f"Mean: {global_mean:.2f}")
    print(f"Median: {np.median(total_errors):.2f}")
    print(f"Std: {global_std:.2f}")

    plt.hist(total_errors, bins=40, color='skyblue', edgecolor='black')
    plt.title(f"Global Error Distribution (Mean: {global_mean:.2f}, Std: {global_std:.2f})")
    plt.xlabel("Error [0.1m]")
    plt.ylabel("Sample Count")
    plt.savefig(os.path.join(main_folder, 'global_statistics.png'))
    plt.show()