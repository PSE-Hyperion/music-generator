import os

from music_generation_lstm.tokenization import tokenizer as t
from music_generation_lstm.midi import parser, writer
from music_generation_lstm.config import DATASETS_MIDI_DIR

# Test to ensure, that the tokenizer works as intented (it did, minus the tempo or whatever it's called)

midi_path = "Tokenize Dis/twinkle.mid"
tokenizer = t.Tokenizer("")
writer.write_midi("test_twinkle", t.detokenize(tokenizer.tokenize(parser.parse_midi(os.path.join(DATASETS_MIDI_DIR, midi_path)))))
