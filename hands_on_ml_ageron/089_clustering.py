import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

mpl.rcParams['font.size'] = 14


def iris_demo():
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    labels = dataset['target_names']
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    cbar = plt.colorbar(ticks=np.unique(y))
    cbar.ax.set_yticklabels(labels)
    plt.show()

    clstr = GaussianMixture(n_components=3, random_state=42)
    y_pred = clstr.fit(X).predict(X)
    mapping = np.array([2, 0, 1])
    y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
    print(accuracy_score(y, y_pred))


def plot_decision_boundaries(clstr, X, resolution=1000):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx1, xx2 = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                           np.linspace(mins[1], maxs[1], resolution))
    y_pred = clstr.predict(np.c_[xx1.ravel(), xx2.ravel()])
    y_pred = y_pred.reshape(xx1.shape)

    plt.contourf(
        y_pred, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap='Pastel2', alpha=0.3
    )
    plt.contour(y_pred, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')


def kmeans_demo():
    def plot_clusters(pos, X, y):
        plt.subplot(pos)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$', rotation=0)

    def generate_data():
        blob_center_list = np.array([
            (0.2, 2.3),
            (-1.5, 2.3),
            (-2.8, 1.8),
            (-2.8, 2.8),
            (-2.8, 1.3),
        ])
        blob_std_list = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        k = len(blob_std_list)
        X, y = datasets.make_blobs(
            n_samples=2000, centers=blob_center_list, cluster_std=blob_std_list, random_state=7
        )
        return X, y, k

    def simple_usage():
        X, y, k = generate_data()
        plt.figure(figsize=(12, 8))
        plot_clusters(121, X, y)

        kmeans = KMeans(n_clusters=k, random_state=42)
        y_pred = kmeans.fit_predict(X)
        print(kmeans.cluster_centers_)
        print(kmeans.transform(X))
        plot_clusters(122, X, y_pred)
        plot_decision_boundaries(kmeans, X)

        plt.show()

    def k_vs_inertia():
        X, y, k = generate_data()
        n_list = np.arange(1, 11, 1)
        kmeans_list = [KMeans(n_clusters=n, random_state=42).fit(X) for n in n_list]
        inertia_list = [kmeans.inertia_ for kmeans in kmeans_list]

        plt.plot(n_list, inertia_list)
        plt.xlabel('K')
        plt.ylabel('inertia_')
        plt.show()

    def k_vs_silhouette():
        X, y, k = generate_data()
        n_list = np.arange(2, 11, 1)
        kmeans_list = [KMeans(n_clusters=n, random_state=42).fit(X) for n in n_list]
        silhouette_list = [silhouette_score(X, kmeans.labels_) for kmeans in kmeans_list]

        plt.plot(n_list, silhouette_list)
        plt.xlabel('K')
        plt.ylabel('silhouette')
        plt.show()

    def preprocessing():
        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        param_grid = {'kmeans__n_clusters': np.arange(10, 110, 10)}
        pipeline = Pipeline([
            ('kmeans', KMeans(random_state=42)),
            ('log_reg', LogisticRegression(random_state=42, multi_class='auto')),
        ])
        search_cv = GridSearchCV(pipeline, param_grid=param_grid, cv=3)
        search_cv.fit(X_train, y_train)
        print(search_cv.best_params_)
        print(search_cv.best_score_)

    def semisupervised():
        def plot_representative(k, X_representative):
            for idx, x in enumerate(X_representative):
                plt.subplot(k // 10, 10, idx + 1)
                plt.imshow(x.reshape(8, 8), cmap='binary', interpolation='bilinear')
                plt.axis('off')
            plt.show()

        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True)

        # train on random 50 samples
        n_labeled = 50
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
        print('random k samples score', log_reg.score(X_test, y_test))

        k = 50
        kmeans = KMeans(n_clusters=k, random_state=42)
        # sample's distance to each cluster center
        X_train_dist = kmeans.fit_transform(X_train)
        # indexes of the closest samples to each center
        representative_idx = np.argmin(X_train_dist, axis=0)
        X_representative = X_train[representative_idx]
        # plot_representative(k, X_representative)

        # manual labeling
        y_representative = np.array([
            4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
            5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
            1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
            6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
            4, 2, 9, 4, 7, 6, 2, 3, 1, 1
        ])
        log_reg.fit(X_representative, y_representative)
        print('semisupervised score', log_reg.score(X_test, y_test))

        # label some percentile of samples
        percentile_closest = 20
        # sample distance to the cluster it belongs
        X_train_cluster_dist = X_train_dist[np.arange(len(X_train)), kmeans.labels_]
        for i in range(k):
            in_cluster = (kmeans.labels_ == i)
            cluster_dist = X_train_cluster_dist[in_cluster]
            cutoff_distance = np.percentile(cluster_dist, percentile_closest)
            above_cutoff = (X_train_cluster_dist > cutoff_distance)
            X_train_cluster_dist[in_cluster & above_cutoff] = -1
        partially_propagated_idx = (X_train_cluster_dist != -1)
        X_train_partially_propagated = X_train[partially_propagated_idx]
        y_train_propagated = np.empty(len(X_train), dtype=np.int32)
        for i in range(k):
            y_train_propagated[kmeans.labels_ == i] = y_representative[i]
        y_train_partially_propagated = y_train_propagated[partially_propagated_idx]
        log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
        print('semisupervised score partially propagation', log_reg.score(X_test, y_test))

    simple_usage()
    preprocessing()
    semisupervised()


def gaussian_demo():
    def generate_data():
        X1, y1 = datasets.make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
        X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
        X2, y2 = datasets.make_blobs(n_samples=250, centers=1, random_state=42)
        X2 = X2 + [6, -8]
        X = np.r_[X1, X2]
        y = np.r_[y1, y2]
        return X, y

    def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
        mins = X.min(axis=0) - 0.1
        maxs = X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                             np.linspace(mins[1], maxs[1], resolution))
        Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z,
                     norm=LogNorm(vmin=1.0, vmax=30.0),
                     levels=np.logspace(0, 2, 12))
        plt.contour(xx, yy, Z,
                    norm=LogNorm(vmin=1.0, vmax=30.0),
                    levels=np.logspace(0, 2, 12),
                    linewidths=1, colors='k')

        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z,
                    linewidths=2, colors='r', linestyles='dashed')

        plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

        plt.xlabel('$x_1$', fontsize=14)
        if show_ylabels:
            plt.ylabel('$x_2$', fontsize=14, rotation=0)
        else:
            plt.tick_params(labelleft='off')

    def outliers(X):
        gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
        gm.fit(X)
        plot_gaussian_mixture(gm, X)
        densities = gm.score_samples(X)
        density_threshold = np.percentile(densities, 4)
        anomalies = X[densities < density_threshold]
        plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
        plt.show()

    def bayes_gauss(X):
        bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
        bgm.fit(X)
        print(np.round(bgm.weights_, 2))
        plot_gaussian_mixture(bgm, X)
        plt.show()

    X, y = generate_data()
    bayes_gauss(X)


gaussian_demo()
