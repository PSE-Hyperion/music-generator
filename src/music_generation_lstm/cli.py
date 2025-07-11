from enum import Enum

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from music_generation_lstm import controller

HELP_INSTRUCTIONS = "the following commands exists:"


class Command(Enum):
    """
    All available commands
    """

    HELP = "-help"
    PROCESS = "-process"
    TRAIN = "-train"
    GENERATE = "-generate"
    SHOW = "-show"
    DELETE = "-delete"
    EXIT = "-exit"


ARGUMENTLENGTH_GENERATE = 3


def handle_help(args: list[str]):
    #   Handles the help command
    #   "-help"
    #
    print(HELP_INSTRUCTIONS)
    for command in Command:
        print(command.name)


def handle_process(args: list[str]):
    #   Handles the process command by calling corresponding controller function
    #   "-process dataset_id processed_id(new)"
    #

    dataset_id = args[0]
    processed_dataset_id = args[1]

    controller.process(dataset_id, processed_dataset_id)


def handle_train(args: list[str]):
    #   Handles the train command by calling corresponding controller function
    #   "-train model_id(new) processed_id"
    #

    model_id = args[0]
    processed_dataset_id = args[1]

    controller.train(model_id, processed_dataset_id)


def handle_generate(args: list[str]):
    #   Handles the generate command by calling corresponding controller function
    #   "-generate model_name input_name result_name(new)"
    #
    if len(args) != ARGUMENTLENGTH_GENERATE:
        print("Incorrect use of the generate command.")
        print("Please use the correct format: -generate [model name] [input name] [desired output name]")

    controller.generate()


def handle_show(args: list[str]):
    #   Handles the show command by calling corresponding controller function
    #   "-show models/raw_datasets/results/processed_datasets (not implemented yet)"
    #
    controller.show()


def handle_delete(args: list[str]):
    #   Handles the delete command for a processed dataset
    #   "-delete"
    processed_dataset_file_id = args[2]
    delete_subject = args[1]

    if delete_subject == "file":
        controller.delete_file(processed_dataset_file_id)
    elif delete_subject == "dataset":
        controller.delete_dataset(processed_dataset_file_id)
    else:
        print(f"Invalid delete subject: {delete_subject}")
    

def handle_exit():
   #   Handles the exit command
   #   "-exit"
   #

   controller.exit()


def parse_command(command: str):
    #   Parses string command to command from Command Enum
    #   Prints Warning, if command doesn't exist
    #

    try:
        return Command(command)
    except ValueError:
        return None


def process_input(input: str):
    #   Split user input into parts and parse command part
    #   Search first part (command) in command map, that maps a command to a handler function
    #   Call handler with arguments, if it exists

    parts = input.split(" ")
    # print("l"+ parts+"p"+len(parts))

    if len(parts) < 0:
        print("Invalid input.Yayyy")
        return

    command = parts[0]
    args = parts[0:]

    command = parse_command(command)

    if command is None:
        print(f"Invalid command. {command} is not a command.")
        return

    length = COMMAND_LENGTH.get(command)

    if length is None:
        print("Command has no length assigned.")
        return

    if length != (len(parts) - 1):
        print(f"Command should get {length} arguments, but got {len(parts) - 1}")
        return

    handler = COMMAND_HANDLERS.get(command)

    if handler is None:
        print("Command has no handler assigned.")
        return

    handler(args)


COMMAND_HANDLERS = {
    Command.PROCESS: handle_process,  # -process dataset_id processed_id(new)
    Command.TRAIN: handle_train,  # -train model_id(new) processed_id
    Command.HELP: handle_help,
    Command.DELETE: handle_delete,
    Command.GENERATE: handle_generate,  # -generate model_id(new) input result_id(new) (not implemented yet)
    Command.SHOW: handle_show,  # -show models/raw_datasets/results/processed_datasets (not implemented yet)
}
COMMAND_LENGTH = {
    Command.PROCESS: 2,  # -process dataset_id processed_id(new)
    Command.TRAIN: 2,  # -train model_id(new) processed_id
    Command.HELP: 0,
    Command.DELETE: 2,  # processed_id
    Command.GENERATE: 2,  # -generate model_id(new) input result_id(new) (not implemented yet)
    Command.SHOW: 0,  # -show models/raw_datasets/results/processed_datasets (not implemented yet)
}


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        #   Only provide completions for the command part
        #   It does this by only considering text before the first space character
        #   We could expand on this by also allowing prompt completion for existing ids, etc

        text = document.text_before_cursor
        parts = text.split(" ")

        if len(parts) <= 1:
            word = parts[0] if parts else ""
            commands = [member.value for member in Command]
            for command in commands:
                if command.startswith(word):
                    yield Completion(command, start_position=-len(word))


def start_session():
    #   Starts the cli loop: input is read, handled and then we repeat (except if exit cmd is called)
    #   Contains a PromptSession for prompt completion in the terminal
    #   Also catches errors from handling input and prints them

    session = PromptSession(completer=CommandCompleter())
    while True:
        try:
            u_input = session.prompt("Music_Generation_LSTM> ")

            if parse_command(u_input.strip()) == Command.EXIT:
                handle_exit()
                break

            process_input(u_input)
        except Exception as e:
            print(f"[ERROR] {e}")
