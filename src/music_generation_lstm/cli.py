# optional seperate cli file

from midi import parser, writer
from tokenization.tokenizer import Tokenizer

from managers.model_management import ModelManager
#from managers.dataset_management import DatasetManager

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion


def handle_u_input(input : str):
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
    except Exception as e:
        print(e)
        return

    try:
        tokenizer = Tokenizer()
        tokens = tokenizer.encode(scores=scores)
        stream = tokenizer.decode(tokens)       # test
        writer.write_midi("test", stream)       # test
    except Exception as e:
        print()
        print(e)
        return




    print("processed")

def handle_train(args : list[str]):
    #   get processed via label
    #   build model
    #   train model
    #   save model
    print("train")

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
    "-p": handle_process,          # -p shortcut_dataset_id kpop110
    "-t": handle_train,              # -t model_id processed_id
    "-g": handle_generate,        # -g model_id input generate_id
    "-s" : handle_show                # -s models/raw_datasets/results/processed_datasets
}

COMMANDS = ["-p", "-t", "-g", "-s", "exit"]

SHORT_CUT_DATASET = {
    "1": "kpop_1_dataset",
    "10": "kpop_10_dataset",
    "110": "kpop_110_dataset"
}



class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split(" ")

        # Only provide completions for the first word
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
