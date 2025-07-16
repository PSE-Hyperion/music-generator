from enum import Enum
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from music_generation_lstm import controller, data_managment

HELP_INSTRUCTIONS = "the following commands exists:"
MIN_DELETE_COMMAND_PARTS = 2
ARG_INDEX_RESULT_ID = 2

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


def handle_help(_args: list[str]):
    #   Handles the help command
    #   "-help"
    #
    logger.info(HELP_INSTRUCTIONS)
    for command in Command:
        logger.info(command.name)


def handle_process(args: list[str]):
    #   Handles the process command by calling corresponding controller function
    #   "-process dataset_id processed_id(new)"
    #

    dataset_id = args[0]
    processed_dataset_id = args[1]

    controller.process(dataset_id, processed_dataset_id)


def handle_train(args: list[str]):
    #   Handles the train command by calling the corresponding controller function
    #   "-train [model_id(new)] [processed_dataset_id] [model_architecture_preset]"
    #

    if len(args) != COMMAND_LENGTHS[Command.TRAIN]:
        logger.error("Incorrect use of the 'train' command.")
        logger.error(
            "Please use the correct format:-train [model name] [processed dataset name] [model architecture preset]"
        )

    model_id = args[0]
    processed_dataset_id = args[1]
    preset_name = (
        args[2] if len(args) > COMMAND_LENGTHS[Command.TRAIN] else "light"
    )  # For when variable length commands are implemented

    controller.train(model_id, processed_dataset_id, preset_name)


def handle_generate(args: list[str]):
    #   Handles the generate command by calling corresponding controller function
    #   Usage: "-generate [model name] [input name] [desired output name]"
    #

    if len(args) != COMMAND_LENGTHS[Command.GENERATE]:
        logger.error("Incorrect use of the 'generate' command.")
        logger.error("Please use the correct format: -generate [model name] [input name] [desired output name]")

    model_name = args[0]
    input_name = args[1]
    output_name = args[2]

    controller.generate(model_name, input_name, output_name)


def handle_show(_args: list[str]):
    #   Handles the show command by calling corresponding controller function
    #   "-show models/raw_datasets/results/processed_datasets (not implemented yet)"
    #
    controller.show()


def handle_delete(args: list[str]):
    #   Handles the delete command for file, dataset, model, processed,
    #   it can delete all instances or one given trough the id
    #   "-delete" file/dataset/model/processed id/all
    id = args[1]
    delete_subject = args[0]

    if delete_subject == "file":
        controller.delete_result(id)
    elif delete_subject == "dataset":
        controller.delete_dataset(id)
    elif delete_subject == "model":
        controller.delete_model(id)
    elif delete_subject == "processed":
        controller.delete_processed(id)
    else:
        logger.info(f"Invalid delete subject: {delete_subject}")


def handle_exit():
    #   Handles the exit command
    #   "-exit"
    #
    controller.exit()


def complete_delete(arg_index, word, parts):
    #   commpletes delete command
    #   first suggestion is what you want to delete(file, dataset, processed, model)
    #   second is the corresponding id or all

    if arg_index == 0:
        for option in ["file", "dataset", "processed", "model"]:
            if option.startswith(word):
                yield Completion(option, start_position=-len(word))

    elif arg_index == 1:
        if "all".startswith(word):  # add 'all' option for deleting everything
            yield Completion("all", start_position=-len(word))

        if len(parts) < MIN_DELETE_COMMAND_PARTS:
            return

        delete_type = parts[1]
        if delete_type == "file":
            id_completion(data_managment.get_existing_result_ids(), word)
        elif delete_type == "dataset":
            id_completion(data_managment.get_existing_dataset_ids(), word)
        elif delete_type == "processed":
            id_completion(data_managment.get_existing_processed_ids(), word)
        elif delete_type == "model":
            id_completion(data_managment.get_existing_model_ids(), word)
        id_sources = {
            "file": data_managment.get_existing_result_ids,
            "dataset": data_managment.get_existing_dataset_ids,
            "processed": data_managment.get_existing_processed_ids,
            "model": data_managment.get_existing_model_ids,
        }

        if delete_type in id_sources:
            yield from id_completion(id_sources[delete_type](), word)


def complete_process(arg_index, word, _parts):
    # completes process command
    # first the all possible dataset-id
    # second the new processed id
    if arg_index == 0:
        yield from id_completion(data_managment.get_existing_dataset_ids(), word)
    if arg_index == 1:
        yield Completion("new_processed_id", start_position=-len(word))


def complete_train(arg_index, word, _parts):
    # completes train command
    # first  the new model id
    # second the all possible processed-id
    if arg_index == 0:
        yield Completion("new_model_id", start_position=-len(word))
    if arg_index == 1:
        yield from id_completion(data_managment.get_existing_processed_ids(), word)


def complete_generate(arg_index, word, _parts):
    # completes generate command
    # first  the all possible model_id input
    # second all possible input
    # third new results_id
    if arg_index == 0:
        yield from id_completion(data_managment.get_existing_model_ids(), word)
    if arg_index == 1:
        yield from id_completion(data_managment.get_existing_processed_ids(), word)
    if arg_index == ARG_INDEX_RESULT_ID:
        yield Completion("new_results_id", start_position=-len(word))


def complete_show(arg_index, word, _parts):
    # completes show command
    if arg_index == 0:
        for option in ["models", "raw_datasets", "results", "processed_datasets"]:
            if option.startswith(word):
                yield Completion(option, start_position=-len(word))


def id_completion(existing_ids, word):
    for id in existing_ids:
        if id.startswith(word):
            yield Completion(id, start_position=-len(word))


def complete_help():
    pass  # should do nothing


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

    length = COMMAND_LENGTHS.get(command)

    if length is None:
        logger.error("Command has no length assigned.")
        return

    if length != (len(parts) - 1):
        logger.info(f"Command should get {length} arguments, but got {len(parts) - 1}")
        return

    handler = COMMAND_HANDLERS.get(command)

    if handler is None:
        logger.error("Command has no handler assigned.")
        return

    handler(args)


COMMAND_HANDLERS = {
    Command.PROCESS: handle_process,  # -process dataset_id processed_id(new)
    Command.TRAIN: handle_train,  # -train model_id(new) processed_id
    Command.HELP: handle_help,  # -help
    Command.DELETE: handle_delete,  # -delete file/dataset/processed/model ids/all
    Command.GENERATE: handle_generate,  # -generate model_id input result_id(new) (not implemented yet)
    Command.SHOW: handle_show,  # -show models/raw_datasets/results/processed_datasets (not implemented yet)
}
COMMAND_LENGTHS = {
    Command.PROCESS: 2,  # -process dataset_id processed_id(new)
    Command.TRAIN: 3,  # -train [model_id(new)] [processed_dataset_id] [model_architecture_preset]
    Command.HELP: 0,
    Command.DELETE: 2,  # file/dataset/processed/model ids/all
    Command.GENERATE: 3,  # -generate model_id input result_id(new) (not implemented yet)
    Command.SHOW: 0,  # -show models/raw_datasets/results/processed_datasets (not implemented yet)
}

COMMAND_COMPLETERS = {
    Command.PROCESS: complete_process,  # dataset_id processed_id(new)
    Command.TRAIN: complete_train,  # model_id, processed_id
    Command.DELETE: complete_delete,  # file/ dataset/processed/model, ids/all
    Command.HELP: complete_help,  # needs no completion
    Command.GENERATE: complete_generate,  # model_id, input, result_id(new)
    Command.SHOW: complete_show,  # not implemented
}


class CommandCompleter(Completer):
    def get_completions(self, document, _event):
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
                    yield Completion(command.value, start_position=-len(current_word))
            return

        if command_enum in COMMAND_COMPLETERS:  # arguments (ids, file, ...)
            if text.endswith(" "):  # " " gedrückt zwischen argumenten
                current_word = ""
                arg_index = len(parts) - 1
            else:
                current_word = parts[-1]
                arg_index = len(parts) - 2

            completer = COMMAND_COMPLETERS[command_enum]
            try:
                yield from completer(arg_index, current_word, parts)
            except Exception as e:
                logger.error(f"[Completion Error] {e}")


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
            logger.error("Something went wrong: %s", e)
            raise Exception(f"[ERROR] {e.__traceback__}") from e
