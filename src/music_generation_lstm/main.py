from midi import parser, writer
from tokenization.tokenizer import Tokenizer


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
    #

    try:
        scores = parser.parse_midi(SHORT_CUT_DATASET[args[0]])
    except Exception as e:
        print(e)

    try:
        tokenizer = Tokenizer()
        tokenizer.encode(scores=scores)
    except Exception as e:
        pass




    print("processed")

def handle_train(args : list[str]):
    print("train")

def handle_generate(args : list[str]):
    print("generate")

COMMAND_HANDLERS = {
    "process": handle_process,          # -p 110 kpop110
    "train": handle_train,
    "generate": handle_generate,

}

SHORT_CUT_DATASET = {
    "1": "kpop_1_d",
    "10": "kpop_10_d",
    "110": "kpop_110_d"
}

if __name__ == "__main__":
    while True:
        u_input = input()
        handle_u_input(u_input)
