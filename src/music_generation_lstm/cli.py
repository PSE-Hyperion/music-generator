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
        scores = parser.parse_midi(SHORT_CUT_DATASET[args[0]])

        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(scores=scores)
        #stream = tokenizer.detokenize(tokens)       # test
        #writer.write_midi("test", stream)           # test

        nums = process.numerize(tokens, tokenizer.token_to_int)
        X, y = process.sequenize(nums)
        X = process.reshape_X(X, tokenizer.num_features)

        DatasetManager.save_tokenized_data(args[1], X, y, (tokenizer.sequence_length, tokenizer.num_features), tokenizer.token_to_int)

    except Exception as e:
        print()
        print(f"[ERROR] {e}")




    print("processed")

def handle_train(args : list[str]):
    #   get processed via id
    #   build model
    #   train model
    #   save model

    try:
        processed_id = args[0]
        model_id = args[1]

        X, y, input_shape, note_to_int = DatasetManager.load_tokenized_data(processed_id)

        model = models.LSTMModel(model_id, input_shape)

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

    print("generate")

def handle_show(args : list[str]):
    #   get model via label
    #   get midi
    #   get start sequence from midi
    #   generate with model using start sequence
    #   write result in folder

    print("generate")

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
    "1": "kpop_1_dataset",
    "10": "kpop_10_dataset",
    "110": "kpop_110_dataset"
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
