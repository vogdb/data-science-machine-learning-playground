import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def generate_X():
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))
    X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
    return X


def pca_demo():
    def compare():
        X = generate_X()
        X_centered = X - np.mean(X, axis=0)
        U, s, V = np.linalg.svd(X_centered)
        X_2D_custom = X_centered.dot(V.T[:, :2])

        pca = PCA(n_components=2)
        X_2D_sklearn = pca.fit_transform(X)
        print(pca.components_.shape)

        # directions can be opposite
        print(np.allclose(X_2D_custom, -X_2D_sklearn))
        print(pca.explained_variance_ratio_)

    def variance():
        X = generate_X()
        pca = PCA()
        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        min_d = np.nonzero(cumsum > .95)[0][0] + 1
        print('minimal D that has > 0.95 of variance is: ', min_d)
        # alternative to the above mess
        pca = PCA(n_components=0.95)
        pca.fit(X)
        print(pca.explained_variance_ratio_)

    def incremental():
        mnist = datasets.fetch_mldata('MNIST original')
        X = mnist['data']
        n_batches = 100
        partial_fit = False
        if partial_fit:
            inc_pca = IncrementalPCA(n_components=154)
            for X_batch in np.array_split(X, n_batches):
                inc_pca.partial_fit(X_batch)

            X_reduced = inc_pca.transform(X)
            X_restored = inc_pca.inverse_transform(X_reduced)
            # X_mm = np.memmap('08_mnist_memmap', dtype='float32', mode='readonly', shape=X.shape)
            # X_mm[:] = X
            # inc_pca = IncrementalPCA(n_components=154, batch_size=X.shape[0] // n_batches)
            # inc_pca.fit(X_mm)
        else:
            rnd_pca = PCA(n_components=X.shape[1], svd_solver='randomized')
            rnd_pca.fit(X)
            cumsum = np.cumsum(rnd_pca.explained_variance_ratio_)
            min_d = np.nonzero(cumsum > 0.95)[0][0] + 1
            print(min_d)

    compare()
    variance()
    incremental()


def kernel_pca_demo():
    X, t = datasets.make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    y = t > 6.9
    pipeline = Pipeline([
        ('kpca', KernelPCA(n_components=2)),
        ('log_reg', LogisticRegression(solver='liblinear')),
    ])
    param_grid = [{
        'kpca__gamma': np.linspace(0.03, 0.05, 10),
        'kpca__kernel': ['rbf', 'sigmoid'],
    }]
    search_cv = GridSearchCV(pipeline, param_grid=param_grid, cv=3, return_train_score=False)
    search_cv.fit(X, y)
    print(search_cv.best_params_)


pca_demo()
kernel_pca_demo()
