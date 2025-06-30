# defines how a given model is trained with the given training data
# might want to add temperature to prediction and other features

import numpy as np

from models.models import BaseModel
from config import TRAINING_EPOCHS, TRAINING_BATCH_SIZE

def temperature():
    pass

def split_X_y(X, y):
    #
    #   Splits X and y into dictionaries of feature-wise arrays for model input and output
    #

    feature_names = ['type', 'pitch', 'duration', 'delta_offset', 'velocity', 'instrument']

    X_dict = {feature_names[i]: X[:, :, i] for i in range(6)}

    y_array = np.array(y, dtype=np.int32)
    y_dict = {feature_names[i]: y_array[:, i] for i in range(6)}

    return X_dict, y_dict

def train_model(model : BaseModel, X, y):
    #   trains the given model
    #
    #

    print(f"Start training {model.name}...", end="\r")
    try:
        X_dict, y_dict = split_X_y(X, y)

        #history =          # optional save history
        model.model.fit(
            X_dict,
            [y_dict[name] for name in ['type', 'pitch', 'duration', 'delta_offset', 'velocity', 'instrument']],
            epochs=TRAINING_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE,
            validation_split=0.1,
            verbose=2   # type: ignore[arg-type]
            # try out 1 and 2. Terminal output is weird with all the different
        )

    except Exception as e:
        raise Exception(f"Training failed: {e}")

    print(f"Finished training {model.name}")

