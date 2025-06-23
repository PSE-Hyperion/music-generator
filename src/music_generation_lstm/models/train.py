# defines how a given model is trained with the given training data
# might want to add temperature to prediction and other features

from models.models import BaseModel
from keras.src.utils import to_categorical
from config import TRAINING_EPOCHS, TRAINING_BATCH_SIZE

def temperature():
    pass

def train_model(model : BaseModel, X, y):
    #   trains the given model
    #
    #

    print(f"Start training {model.name}...", end="\r")

    try:
        num_classes = model.input_shape[1]

        X = to_categorical(X, num_classes)
        y = to_categorical(y, num_classes)

        #history =          # optional save history
        model.model.fit(x=X, y=y, epochs=TRAINING_EPOCHS, batch_size=TRAINING_BATCH_SIZE)
    except Exception as e:
        raise Exception(f"Training failed: {e}")

    print(f"Finished training {model.name}")
