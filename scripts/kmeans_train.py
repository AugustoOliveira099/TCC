from open_ai.kmeans import kmeans
from open_ai.kmeans import test_kmeans_model

if __name__ == '__main__':
    kmeans.train_model()
    test_kmeans_model.test_model()
