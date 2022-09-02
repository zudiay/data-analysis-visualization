import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.manifold import TSNE


# Plot 10 sample digit images per digit class.
def plot_ten_sample_digits(train_x, train_y):
    f = plt.figure(figsize=(10, 10))
    for digit in range(0, 10):
        count, i = 0, 0
        while count < 10:
            if train_y[i] == digit:
                f.add_subplot(10, 10, digit * 10 + count + 1)
                plt.imshow(train_x[i].reshape([28, 28]), cmap="Greys")
                plt.axis("off")
                count += 1
            i += 1
    plt.savefig("digit_images.png")


# Plot the mean image
def plot_mean_image(train_x):
    f = plt.figure()
    mean = np.average(train_x, axis=0)
    plt.imshow(mean.reshape([28, 28]), cmap="Greys")
    plt.savefig('mean_image.png')


# Plot eigenvalues
def plot_50_eigenvalues(value_list):
    f = plt.figure()
    plt.plot(value_list[:50], marker='o', linewidth=0.5, markersize=3)
    plt.savefig('eigenvalues.png')


# Plot largest 100 eigenvectors
def plot_100_eigenvectors(vector_list):
    f = plt.figure(figsize=(10, 10))
    for i in range(0, 100):
        f.add_subplot(10, 10, i + 1)
        plt.imshow(vector_list[i].reshape([28, 28]), cmap="gray")
        plt.axis("off")
    plt.savefig("eigenvectors.png")


# Using the MNIST test set, reduce the dimensionality of features to two for PCA visualization.
def plot_pca(eigenvectors, test_x, test_y):
    projection_1, projection_2 = test_x.dot(eigenvectors.T[0]), test_x.dot(eigenvectors.T[1])
    res = pd.DataFrame({'Projection 1': projection_1, 'Projection 2': projection_2, 'Digit': test_y})
    sn.FacetGrid(res, hue="Digit", height=10).map(plt.scatter, 'Projection 1', 'Projection 2')
    plt.legend()
    plt.savefig("digits_2D_pca.png")


# Use t-SNE to visualize MNIST test set
def plot_tsne(test_x, test_y):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_x = tsne.fit_transform(test_x)
    res = pd.DataFrame({'Projection 1': tsne_x.T[0], 'Projection 2': tsne_x.T[1], 'Digit': test_y})
    sn.FacetGrid(res, hue='Digit', height=10).map(plt.scatter, 'Projection 1', 'Projection 2')
    plt.legend()
    plt.savefig("digits_2D_tsne.png")


def app():
    # read the datasets from csv
    df_train = pd.read_csv('mnist_train.csv')
    train_x = np.asarray(df_train.iloc[:, 1:])
    train_y = np.asarray(df_train.iloc[:, 0])
    train_x = train_x / 255

    df_test = pd.read_csv('mnist_test.csv')
    test_x = np.asarray(df_test.iloc[:, 1:])
    test_y = np.asarray(df_test.iloc[:, 0])
    test_x = test_x / 255

    # create covariance matrix, read eigenvectors and eigenvalues into lists
    covariance_matrix = np.cov(train_x.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    value_list, vector_list = [], []
    for i in range(len(eigenvalues)):
        value_list.append(eigenvalues[i])
        vector_list.append(eigenvectors[:, i])

    # plot the required images
    plot_ten_sample_digits(train_x, train_y)
    plot_mean_image(train_x)
    plot_100_eigenvectors(vector_list)
    plot_50_eigenvalues(value_list)
    plot_pca(eigenvectors, test_x, test_y)
    plot_tsne(test_x, test_y)


if __name__ == '__main__':
    app()
