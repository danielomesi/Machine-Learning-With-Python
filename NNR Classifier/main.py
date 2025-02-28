import time
import json
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from typing import List
from sklearn.metrics import accuracy_score
import numpy as np

def seperate_features_and_target(df, name_of_target : str):
    target_df = df[name_of_target]
    features_df = df.drop([name_of_target], axis=1)

    return features_df, target_df

def scale_df(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_features = scaler.transform(df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=df.columns, index=df.index)

    return scaled_features_df

def get_accuracy(predictions, actual_targets):
    """"
    returns rate of successful predictions
    """
    correct_predictions = np.sum(predictions == actual_targets)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy

def predict(scaled_features_with_known_target_df, target_df, scaled_features_to_predict_df, radius):
    # retrieve the options of labels
    target_set = set(target_df)
    predictions = []

    scaled_features_with_known_target_array = scaled_features_with_known_target_df.to_numpy()
    scaled_features_to_predict_array = scaled_features_to_predict_df.to_numpy()

    # calculate the distances between each vector in the train data and each vector in the data to predict
    distances = np.linalg.norm(scaled_features_to_predict_array[:, np.newaxis, :] - scaled_features_with_known_target_array, axis=2)

    for idx, vector_to_predict in enumerate(scaled_features_to_predict_array):
        # reset map
        target_map = {value: 0 for value in target_set}
        # take the vectors inside my radius to be my neighbors
        neighbor_indices = np.where(distances[idx] <= radius)[0]

        # add to the specific label the neighbour that belongs to it
        for i in neighbor_indices:
            target_map[target_df.iloc[i]] += 1

        # decide which is the best label according to the label with the most neighbours in it
        max_key = max(target_map, key=target_map.get)
        predictions.append(max_key)

    return predictions

def approximate_average_euclidean_distance(df, sample_size : int):
    """"
    The function is returning the mean of the distances between each pair in a sampled size part of the data frame
    """
    vectors_array = df.to_numpy()
    random_pairs_indices = np.random.choice(len(vectors_array), size=(sample_size, 2))
    sampled_pairs = vectors_array[random_pairs_indices]
    sampled_distances = np.linalg.norm(sampled_pairs[:, 0, :] - sampled_pairs[:, 1, :], axis=1)
    approx_avg_distance = np.mean(sampled_distances)

    return approx_avg_distance

def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # read the data from csv to dataframes
    data_trn_df = pd.read_csv(data_trn)
    data_vld_df = pd.read_csv(data_vld)

    # separate to features and target class
    features_trn_df, target_trn_df = seperate_features_and_target(data_trn_df, 'class')
    features_vld_df, target_vld_df = seperate_features_and_target(data_vld_df, 'class')

    # scaling the features of the training data and the validation data
    scaled_features_trn_df = scale_df(features_trn_df)
    scaled_features_vld_df = scale_df(features_vld_df)
    scaled_features_tst_df = scale_df(df_tst)

    # calculate the mean of the distance between sampled vectors out of the train data
    sample_size = int(len(scaled_features_trn_df)/10)
    approximate_average_of_distance = approximate_average_euclidean_distance(scaled_features_trn_df, sample_size)

    # determine the range of the checked radiuses
    min_radius = approximate_average_of_distance / 10
    max_radius = approximate_average_of_distance / 2
    num_of_radiuses = 25
    step_between_radiuses = (max_radius - min_radius) / num_of_radiuses

    # set  up all the radius values in an array
    radius_values = np.arange(min_radius, max_radius, step_between_radiuses.tolist())
    best_radius = min_radius
    max_accuracy = 0
    current_round = 1

    # iterate the radiuses and check for the accuracy that corresponds to each iterated radius
    for radius in radius_values:
        print(f"Round {current_round} out of {num_of_radiuses} - Radius: {radius} in range [{min_radius}, {max_radius}]")
        print("Calculating accuracy...")
        predicted = predict(scaled_features_trn_df, target_trn_df, scaled_features_vld_df, radius)
        accuracy = get_accuracy(predicted, target_vld_df)
        print("Reached accuracy = " + str(accuracy))
        if accuracy > max_accuracy:
            best_radius = radius
            max_accuracy = accuracy
        current_round += 1

    print("min radius: " + str(min_radius))
    print("max radius: " + str(max_radius))
    print("best radius: "+str(best_radius)+" with accuracy of "+str(max_accuracy))

    # predict the test data with the best radius found
    predictions = predict(scaled_features_trn_df, target_trn_df, scaled_features_tst_df, best_radius)
    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
