import numpy as np
from sklearn.decomposition import PCA

class CustomKMeans:
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
    
    def fit(self, X):
        np.random.seed(self.random_state)
        # Redução de dimensionalidade usando PCA
        pca = PCA(n_components=min(self.n_clusters, X.shape[1]))
        X_reduced = pca.fit_transform(X)
        
        # Inicializar centróides
        centroids = X_reduced[np.random.choice(X_reduced.shape[0], self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            # Atribuir clusters
            labels = self._assign_clusters(X_reduced, centroids)
            # Atualizar centróides
            new_centroids = np.array([X_reduced[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            # Verificar convergência
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < self.tol):
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels_ = labels
    
    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("O modelo precisa ser ajustado antes de fazer previsões.")
        
        # Redução de dimensionalidade usando PCA
        pca = PCA(n_components=self.centroids.shape[1])
        X_reduced = pca.fit_transform(X)
        
        # Atribuir clusters com base nos centróides ajustados
        labels = self._assign_clusters(X_reduced, self.centroids)
        return labels
