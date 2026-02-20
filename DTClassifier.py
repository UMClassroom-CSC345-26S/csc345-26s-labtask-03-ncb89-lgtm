#----Eliminating warnings from scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree

#--------------------------------------------------------------------------------------------------
def get_data(filepath):

#----Read the cleaned and normalized dataset from Labtask 2
    dataset = pd.read_csv(filepath)
#----Return the feature matrix (Volume, Doors) and the true style labels
    features = dataset[['Volume_Normalized', 'Doors_Normalized']].values
    styles   = dataset['Style'].values
    return features, styles

#--------------------------------------------------------------------------------------------------
def split_data(features, styles, test_fraction, random_seed):

#----Randomly split into 80% training and 20% testing sets
    features_train, features_test, styles_train, styles_test = train_test_split(
        features, styles,
        test_size=test_fraction,
        random_state=random_seed
    )
    return features_train, features_test, styles_train, styles_test

#--------------------------------------------------------------------------------------------------
def train_decision_tree(features_train, styles_train):

#----Build a decision tree classifier on the training data
    model = DecisionTreeClassifier(random_state=0)
    model.fit(features_train, styles_train)
    return model

#--------------------------------------------------------------------------------------------------
def save_tree_image(model, feature_names, class_names, output_path):

#----Save a visual representation of the decision tree to a PNG file
    figure, plot_area = plt.subplots(figsize=(20, 10))
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        ax=plot_area,
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

#--------------------------------------------------------------------------------------------------
def save_tree_cars(features_test, styles_test, predicted_styles, accuracy, output_path):

#----Build a dataframe of test-set cars with their true and predicted styles
    results = pd.DataFrame({
        'Volume':         features_test[:, 0],
        'Doors':          features_test[:, 1],
        'Style':          styles_test,
        'PredictedStyle': predicted_styles
    })
#----Append one summary row at the bottom showing the overall prediction accuracy
    accuracy_row = pd.DataFrame([{
        'Volume':         'Accuracy',
        'Doors':          '',
        'Style':          '',
        'PredictedStyle': round(accuracy, 4)
    }])
    results = pd.concat([results, accuracy_row], ignore_index=True)
    results.to_csv(output_path, index=False)

#--------------------------------------------------------------------------------------------------
def compute_accuracy(styles_test, predicted_styles):

#----Accuracy = number of correct predictions / total number of test cars
    correct = np.sum(styles_test == predicted_styles)
    accuracy = correct / len(styles_test)
    return accuracy

#--------------------------------------------------------------------------------------------------
def main():

    test_fraction  = 0.20
    random_seed    = 42
    data_path      = 'CleanedAndNormalizedFromLab2.csv'
    tree_image_path = 'TreeCars.png'
    tree_cars_path  = 'TreeCars.csv'
    feature_names   = ['Volume_Normalized', 'Doors_Normalized']

#----Load features and true style labels
    features, styles = get_data(data_path)

#----Split into training (80%) and testing (20%) sets
    features_train, features_test, styles_train, styles_test = split_data(
        features, styles, test_fraction, random_seed
    )
    print(f"Training set size: {len(features_train)}, Testing set size: {len(features_test)}")

#----Train the decision tree on the training set
    model = train_decision_tree(features_train, styles_train)

#----Save the decision tree diagram to a PNG
    class_names = sorted(np.unique(styles).tolist())
    save_tree_image(model, feature_names, class_names, tree_image_path)
    print(f"Saved {tree_image_path}")

#----Predict styles for the test set cars
    predicted_styles = model.predict(features_test)

#----Calculate prediction accuracy on the test set
    accuracy = compute_accuracy(styles_test, predicted_styles)
    print(f"Prediction accuracy: {round(accuracy, 4)}")

#----Save test-set predictions and accuracy to CSV
    save_tree_cars(features_test, styles_test, predicted_styles, accuracy, tree_cars_path)
    print(f"Saved {tree_cars_path}")

#--------------------------------------------------------------------------------------------------
main()
