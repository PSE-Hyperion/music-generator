import controller
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
    dataset_id = args[0]
    processed_dataset_id = args[1]

    controller.process(dataset_id, processed_dataset_id)


def handle_train(args : list[str]):                 # TRAIN DOESNT WORK NOW, SINCE THE PROCESSED DATA IS SAVED DIFFERENTLY
    try:
        model_id = args[0]
        processed_dataset_id = args[1]

        controller.train(model_id, processed_dataset_id)
    except Exception as e:
        print(f"[ERROR] {e}")


def handle_generate(args : list[str]):
    controller.generate()

def handle_show(args : list[str]):
    controller.show()

def handle_exit():
    controller.exit()

COMMAND_HANDLERS = {
    "-process": handle_process,          # -p shortcut_dataset_id processed_id(new)
    "-train": handle_train,              # -t model_id(new) processed_id
    "-generate": handle_generate,        # -g model_id input generate_id
    "-show" : handle_show                # -s models/raw_datasets/results/processed_datasets
}

COMMANDS = ["-process", "-train", "-generate", "-show", "exit"]




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
        except Exception as e:
            print(f"[ERROR] {e}")
