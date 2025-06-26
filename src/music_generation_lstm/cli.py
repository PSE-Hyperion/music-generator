# optional seperate cli file

import models.models as models
from models import train

from midi import parser, writer
from processing import process
from tokenization.tokenizer import Tokenizer

from managers.model_management import ModelManager
from managers.dataset_management import DatasetManager

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion


def handle_u_input(input : str):
    #   split user input into parts
    #   search first part (command) in command map, that maps a command to a handler function
    #

    parts = input.split(" ")
    if len(parts) < 2:
        print("Invalid input.")
        return
    command = parts[0]
    args = parts[1:]

    handler = COMMAND_HANDLERS.get(command, None)

    if handler is None:
        print("Invalid command.")
        return

    handler(args)


def handle_process(args : list[str]):
    #   parses midi file(s) to music21.stream.Score
    #   tokenize score(s)
    #   numerize tokens
    #   save processed data (ready for training data)

    try:
        dataset_id = SHORT_CUT_DATASET[args[0]]
        processed_dataset_id = args[1]

        midi_paths = parser.get_midi_paths_from_dataset(dataset_id)
        tokenizer = Tokenizer(processed_dataset_id)

        total = len(midi_paths)

        for index, midi_path in enumerate(midi_paths, start=1):    # seperate load and processing, share vocab data in shared tokenizer object
            try:
                print(f"[PROGRESS] Processing ({index}/{total})")
                score = parser.parse_midi(midi_path)

                embedded_token_events = tokenizer.tokenize(score)   # might be handled now

                embedded_numeric_events = process.numerize(embedded_token_events, tokenizer)   # might be handled now
                X, y = process.sequenize(embedded_numeric_events)   # might be handled now
                X = process.reshape_X(X)

                DatasetManager.save_processed_data(processed_dataset_id, midi_path, X, y, tokenizer) # might be handled now
            except Exception as e:
                print(f"[ERROR] {e}")       # WARNING, when song is too short for sequence, the maps are still updated to contain the tokens of the song
        tokenizer.save_maps()

    except Exception as e:
        print()
        print(f"[ERROR] {e}")


def handle_train(args : list[str]):                 # TRAIN DOESNT WORK NOW, SINCE THE PROCESSED DATA IS SAVED DIFFERENTLY
    #   get processed via id
    #   build model
    #   train model
    #   save model

    try:
        model_id = args[0]
        processed_dataset_id = args[1]

        X, y, input_shape, map_id = DatasetManager.load_tokenized_data(processed_dataset_id)

        # uh uh, stinky: update asap
        import os
        import json
        token_maps_dir = os.path.join("data/token_maps", map_id)
        with open(os.path.join(token_maps_dir, "type_map")) as f:
            type_map = json.load(f)
        with open(os.path.join(token_maps_dir, "pitch_map")) as f:
            pitch_map = json.load(f)
        with open(os.path.join(token_maps_dir, "duration_map")) as f:
            duration_map = json.load(f)
        with open(os.path.join(token_maps_dir, "delta_offset_map")) as f:
            delta_offset_map = json.load(f)
        with open(os.path.join(token_maps_dir, "velocity_map")) as f:
            velocity_map = json.load(f)
        with open(os.path.join(token_maps_dir, "instrument_map")) as f:
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

    except Exception as e:
        print(f"[ERROR] {e}")


def handle_generate(args : list[str]):
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    return
    ModelManager()
    writer

    print("generate")

def handle_show(args : list[str]):
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("show")

def handle_exit():
    print("You've exited the program.")

COMMAND_HANDLERS = {
    "-process": handle_process,          # -p shortcut_dataset_id kpop110
    "-train": handle_train,              # -t model_id processed_id
    "-generate": handle_generate,        # -g model_id input generate_id
    "-show" : handle_show                # -s models/raw_datasets/results/processed_datasets
}

COMMANDS = ["-process", "-train", "-generate", "-show", "exit"]

SHORT_CUT_DATASET = {
    "k1": "kpop_1_dataset",
    "k2": "kpop_10_dataset",
    "k3": "kpop_110_dataset",
    "m1": "maestro_1_dataset",
    "m2": "maestro_10_dataset",
    "m3": "maestro_50_dataset"
}



class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split(" ")

        # Only provide completions for the command part             (maybe extend later to also provide completion for dataset and model ids)
        if len(parts) <= 1:
            word = parts[0] if parts else ""
            for cmd in COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))


def start_session():
    session = PromptSession(completer=CommandCompleter())
    while True:
        try:
            u_input = session.prompt("Music_Generation_LSTM> ")
            if u_input.strip() == "exit":
                handle_exit()
                break
            handle_u_input(u_input)
        except KeyboardInterrupt:
            break
