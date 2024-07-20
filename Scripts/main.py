from data_preprocessing import preprocess_data
from train_models import get_best_model
import joblib
import keras


def main():
    train_df, test_df = preprocess_data()
    best_model = get_best_model(train_df=train_df, test_df=test_df)

    if isinstance(best_model, keras.models.Sequential):
        best_model.save("../pickle_files/model.keras")
    else:
        with open("../pickle_files/model.pkl", "wb") as f:
            joblib.dump(best_model, f)

    return best_model


main()
