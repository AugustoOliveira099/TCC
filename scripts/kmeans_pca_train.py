from open_ai.kmeans_pca import kmeans_with_pca
from open_ai.kmeans_pca import test_kmeans_pca_model

if __name__ == '__main__':
    kmeans_with_pca.train_model()
    test_kmeans_pca_model.test_model()
