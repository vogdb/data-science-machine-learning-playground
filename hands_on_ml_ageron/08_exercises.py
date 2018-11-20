import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def exercise9():
    mnist = datasets.fetch_mldata('MNIST original')
    X = mnist['data']
    y = mnist['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

    pca_worse = False
    if pca_worse:
        clf = RandomForestClassifier(n_estimators=50, max_depth=8)
    else:
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print('plain time: {}, accuracy: {}'.format(end - start, clf.score(X_test, y_test)))

    pca = PCA(n_components=154, svd_solver='randomized')
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    start = time.time()
    clf.fit(X_train_reduced, y_train)
    end = time.time()
    print('pca time: {}, accuracy: {}'.format(end - start, clf.score(X_test_reduced, y_test)))


def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    X_normalized = MinMaxScaler().fit_transform(X)
    neighbors = np.array([[10., 10.]])
    plt.figure(figsize=figsize)
    cmap = matplotlib.cm.get_cmap('jet')
    # digits = np.unique(y)
    # for digit in digits:
    #     plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=cmap(digit / 9))
    plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={'weight': 'bold', 'size': 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap='binary'), image_coord)
                ax.add_artist(imagebox)


def exercise10():
    mnist = datasets.fetch_mldata('MNIST original')
    m = 5000
    # m = 1000 # MDS
    idx = np.random.permutation(60000)[:m]
    X = mnist['data'][idx]
    y = mnist['target'][idx]

    # reduct = TSNE(n_components=2, random_state=42)
    # reduct = PCA(n_components=2, random_state=42)
    # reduct = LocallyLinearEmbedding(n_components=2, random_state=42)
    # reduct = MDS(n_components=2, random_state=42)
    reduct = Pipeline([
        ('pca_to_speed_up', PCA(n_components=.95, random_state=42)),
        ('tsne', TSNE(n_components=2, random_state=42)),
    ])
    X_reduced = reduct.fit_transform(X)

    plot_digits(X_reduced, y, images=X, min_distance=0.07)
    plt.show()


exercise10()
