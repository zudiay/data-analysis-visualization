# k-means algorithm

Implementation of the k-means clustering algorithm from scratch.

1. A 2D toy/simple dataset is generated that is suitable for clustering.
2. k-Means algorithm is applied to the dataset and moving cluster centers are plotted. (Two different k values are used):
    1) for the first three iterations and 
    2) for the last iteration.
3. Objective function vs iteration count is plotted for all iterations.
4. Final clustering is compared with the output obtained by the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
5. A method is implemented to find the best k automatically. Final cluster centers found by this method on the dataset is shown. 


Python version: Python 3.9.5

Open the terminal in the code folder.
Run the following command to install the requirements
````pip3 install -r requirements.txt````

Run the following command:
````python3 assignment1.py````
This command will run the code for the convex dataset. 

If you want to see the results for the non-convex dataset, run the following command:
````python3 assignment1.py False````



<i>Developed for CMPE 481 SP.TP.IN DATA ANALYSIS AND VISUALIZATION Course, Bogazici University, Fall 2021.<i>
