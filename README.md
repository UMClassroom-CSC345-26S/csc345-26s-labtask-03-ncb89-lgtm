[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/TEt0gn-5)
# Find the Car Style

This labtask has two parts. Both parts use the cleaned and normalized data from Labtask #2.
- Can a K-Means analysis discover the styles of cars from their interior volume and number of doors?
  Write Python code called `KMeansClassifier.ipynb`/`KMeansClassifier.py` (depending how you 
  implement it) that uses K-Means to cluster all the cars into 5 clusters based on their
  interior volume and number of doors.
  For each cluster find the majority style, one of `Sedan`, `SUV`, `Jeep`, `Pickup`, `Van`,
  which is then the style of that cluster.
  Create `ClusterCars.csv` containing the `Volume`, `Doors`, `Style`, and the `ClusterStyle`,
  for each car.
  Compute the accuracy of each cluster as the number of cars that do have that cluster's style,
  divided by the size of the cluster.
  Create `ClusterAccuracy.csv` with five rows (one row for each style), with columns 
  `ClusterStyle`, `SizeOfCluster`, `Accuracy`.
  Submit `ClusterCars.csv`, `ClusterAccuracy.csv`, and 
  `KMeansClassifier.ipynb`/`KMeansClassifier.py`. (2.0%)
- Can a Decision Tree be used to predict the style of a car from its interior volume and number of doors?
  Randomly split the data set into a training set of 80% and a testing set of 20%.
  Write Python code called `DTClassifier.ipynb`/`DTClassifier.py` (depending how you implement it)
  that builds a decision tree for the training set, and predicts the styles of the cars in the
  testing set.
  Save the decision tree in file called `TreeCars.png`.
  Create `TreeCars.csv` containing the `Volume`, `Doors`, `Style`, and the `PredictedStyle`,
  for each car in the testing set.
  Add a row at the bottom that gives the accuracy of the predictions.
  Submit `TreeCars.png`, `TreeCars.csv`, and `DTClassifier.ipynb`/`DTClassifier.py`. (2.0%)
