import models.models as models
from processing import process as p
from midi import parser, writer
from tokenization.tokenizer import Tokenizer
from managers.model_management import ModelManager
from managers.dataset_management import DatasetManager
from processing import process as p

def process(dataset_id: str, processed_dataset_id: str):
    #   parses midi file(s) to music21.stream.Score
    #   tokenize score(s)
    #   numerize tokens
    #   save processed data (ready for training data)


    midi_paths = parser.get_midi_paths_from_dataset(dataset_id)
    tokenizer = Tokenizer(processed_dataset_id)

    total = len(midi_paths)

    for index, midi_path in enumerate(midi_paths, start=1):    # seperate load and processing, share vocab data in shared tokenizer object
        print(f"[PROGRESS] Processing ({index}/{total})")
        score = parser.parse_midi(midi_path)

        embedded_token_events = tokenizer.tokenize(score)   # might be handled now

        embedded_numeric_events = p.numerize(embedded_token_events, tokenizer)   # might be handled now
        X, y = p.sequenize(embedded_numeric_events)   # might be handled now
        X = p.reshape_X(X)

        DatasetManager.save_processed_data(processed_dataset_id, midi_path, X, y, tokenizer) # might be handled now
    tokenizer.save_maps()


def train(model_id: str, processed_dataset_id: str):                 # TRAIN DOESNT WORK NOW, SINCE THE PROCESSED DATA IS SAVED DIFFERENTLY
    #   get processed via id
    #   build model
    #   train model
    #   save model

    print("DOESNT WORK YET")
    # need to load processed data correctly to work
    return

    X, y, input_shape, map_id = DatasetManager.load_tokenized_data(processed_dataset_id)

    # uh uh, stinky: update asap
    import os
    import json
    token_maps_dir = os.path.join("data/token_maps", map_id)
    with open(os.path.join(token_maps_dir, "type_map.json")) as f:
        type_map = json.load(f)
    with open(os.path.join(token_maps_dir, "pitch_map.json")) as f:
        pitch_map = json.load(f)
    with open(os.path.join(token_maps_dir, "duration_map.json")) as f:
        duration_map = json.load(f)
    with open(os.path.join(token_maps_dir, "delta_offset_map.json")) as f:
        delta_offset_map = json.load(f)
    with open(os.path.join(token_maps_dir, "velocity_map.json")) as f:
        velocity_map = json.load(f)
    with open(os.path.join(token_maps_dir, "instrument_map.json")) as f:
        instrument_map = json.load(f)

    vocab_sizes = {
        'type': len(type_map),
        'pitch': len(pitch_map),
        'duration': len(duration_map),
        'delta_offset': len(delta_offset_map),
        'velocity': len(velocity_map),
        'instrument': len(instrument_map),
    }

    model = models.LSTMModel(model_id, input_shape)
    model.build(vocab_sizes=vocab_sizes)

    train.train_model(model, X, y)

    # save model


def generate():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder


    print("generate")


def show():
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("show")


def exit():
    print("You've exited the program.")
