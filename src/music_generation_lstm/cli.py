import logging
from enum import Enum

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from music_generation_lstm import controller
from music_generation_lstm import data_managment


HELP_INSTRUCTIONS = "the following commands exists:"

logger = logging.getLogger(__name__)

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
    #   Usage: "-generate [model name] [input name] [desired output name]"
    #
    
    if len(args) != ARGUMENTLENGTH_GENERATE:
        print("Incorrect use of the generate command.")
        print("Please use the correct format: -generate [model name] [input name] [desired output name]")
    if len(args) != ARGUMENTLENGTH_GENERATE:
        logger.error("Incorrect use of the generate command.")
        logger.error("Please use the correct format: -generate [model name] [input name] [desired output name]")

    model_name = args[0]
    input_name = args[1]
    output_name = args[2]

    controller.generate(model_name, input_name, output_name)


def handle_show(args: list[str]):
    #   Handles the show command by calling corresponding controller function
    #   "-show models/raw_datasets/results/processed_datasets (not implemented yet)"
    #
    controller.show()


def handle_delete(args: list[str]):
    #   Handles the delete command for a processed dataset
    #   "-delete"
    processed_dataset_file_id = args[1]
    delete_subject = args[0]

    if delete_subject == "file":
        controller.delete_result(processed_dataset_file_id)
    elif delete_subject == "dataset":
        controller.delete_dataset(processed_dataset_file_id)
    else:
        print(f"Invalid delete subject: {delete_subject}")
    

def handle_exit():
   #   Handles the exit command
   #   "-exit"
   #

   controller.exit()

"""
def complete_delete(arg_index, word, parts):
    if arg_index == 0:
        for option in ["file", "dataset"]:
            if option.startswith(word):
                yield Completion(option, start_position=-len(word))
                
    if arg_index == 1:
        delete_type = parts[1]
        if delete_type == "file":
            for fid in data_managment.get_existing_result_ids():
                if fid.startswith(word):
                    yield Completion(fid, start_position=-len(word))
        elif delete_type == "dataset":
            for did in data_managment.get_existing_dataset_ids():
                if did.startswith(word):
                    yield Completion(did, start_position=-len(word))
    

def complete_process():
    print("handle")
def complete_train():
    print("handle")
def complete_help():
    print("handle")
    # do nothing
def complete_generate():
    print("handle")
def complete_show():
    print("handle")
"""

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

    if len(parts) < 0:
        logger.error("Invalid input.")
        return

    command = parts[0]
    args = parts[1:]

    command = parse_command(command)

    if command is None:
        logger.error("Invalid command. %s is not a command.", parts[0])
        return

    length = COMMAND_LENGTH.get(command)

    if length is None:
        logger.error("Command has no length assigned.")
        return

    if length != (len(parts) - 1):
        print(f"Command should get {length} arguments, but got {len(parts) - 1}")
        return

    handler = COMMAND_HANDLERS.get(command)

    if handler is None:
        logger.error("Command has no handler assigned.")
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
    Command.GENERATE: 3,  # -generate model_id(new) input result_id(new) (not implemented yet)
    Command.SHOW: 0,  # -show models/raw_datasets/results/processed_datasets (not implemented yet)
}

"""COMMAND_COMMPLETER = {
    Command.PROCESS: complete_process,  # dataset_id processed_id(new)
    Command.TRAIN: complete_train,  # model_id, processed_id
    Command.DELETE: complete_delete, # file/ dataset, ids
    Command.HELP: complete_help,  # needs no completion
    Command.GENERATE: complete_generate,  # not implemented
    Command.SHOW: complete_show #not implemented
}"""


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split()

        if not parts:
            for command in Command:
                yield Completion(command.value, start_position=0)
            return

        command = parts[0]
        command_enum = parse_command(command)

        
        if len(parts) == 1 and not text.endswith(" "):  # command name(-delete, -train, ...)
            current_word = parts[0]
            for command in Command:
                if command.value.startswith(current_word):
                    yield Completion(command.value, start_position = -len(current_word))
            return

        
        """if command_enum in COMMAND_COMMPLETER: # arguments (ids, file, ...)
            if text.endswith(" "): # " " gedrückt zwischen argumenten
                current_word = ""
                arg_index = len(parts) - 1
            else:
                current_word = parts[-1]
                arg_index = len(parts) - 2

            completer = COMMAND_COMMPLETER[command_enum]
            try:
                yield from completer(arg_index, current_word, parts)
            except Exception as e:
                print(f"[Completion Error] {e}")"""



        
        


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
            logger.error("%s", e)
