from xgboost import initial_model
from xgboost import final_model

if __name__ == '__main__':
    initial_model.train_model()
    final_model.train_model()
