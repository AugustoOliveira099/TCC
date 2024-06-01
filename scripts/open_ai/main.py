import kmeans
import open_ai_api
import training_model
import naming_clusters

if __name__ == '__main__':
    open_ai_api.main() # Run this will take a long time and cost money
    kmeans.main()
    training_model.main()
    naming_clusters.main() # Run this cost money
