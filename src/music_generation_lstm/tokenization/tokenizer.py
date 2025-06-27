
import json

from config import SEQUENCE_LENGTH, QUANTIZATION_RESOLUTION, TOKEN_MAPS_DIR
import os
from typing import List, Dict, Any, Tuple
from mido import MidiFile, MidiTrack, Message, MetaMessage






def quantize(value, resolution : float = 1/8) -> float:
    return round(value / resolution) * resolution

def _read_and_merge_events(midi: MidiFile) -> Tuple[List[Dict], int]:
        """
        Merge tempo, time‐signature, note, and pedal (control change 64,66,67) events
        from all tracks into a global, time‐sorted list with absolute ticks.
        """
        ppq = midi.ticks_per_beat
        merged: List[Dict] = []

        # Meta events (tempo, time_signature) from track 0
        abs_tick = 0
        for msg in midi.tracks[0]:
            abs_tick += msg.time
            if msg.is_meta and msg.type in ('set_tempo', 'time_signature'):
                entry = {'abs_tick': abs_tick, 'type': msg.type}
                if msg.type == 'set_tempo':
                    entry['tempo'] = msg.tempo
                else:
                    entry['numerator'] = msg.numerator
                    entry['denominator'] = msg.denominator
                merged.append(entry)

        # Note and pedal events from every track
        for track in midi.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                # Pedal CC: 64 = sustain, 66 = sostenuto, 67 = soft
                if msg.type == 'control_change' and msg.control in (64, 66, 67):
                    merged.append({
                        'abs_tick': abs_tick,
                        'type': 'control_change',
                        'control': msg.control,
                        'value': msg.value
                    })
                # Note on
                elif msg.type == 'note_on' and msg.velocity > 0:
                    merged.append({
                        'abs_tick': abs_tick,
                        'type': 'note_on',
                        'note': msg.note,
                        'velocity': msg.velocity
                    })
                # Note off (or on with zero velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    merged.append({
                        'abs_tick': abs_tick,
                        'type': 'note_off',
                        'note': msg.note,
                        'velocity': 0
                    })

        # Sort globally by absolute tick
        merged.sort(key=lambda e: e['abs_tick'])
        return merged, ppq

def note_on_event(event : Dict[Any, Any]) -> List[str]:
    return ["NOTE_ON", str(event["note"]), str(event["velocity"])]

def note_off_event(event : Dict[Any, Any]) -> List[str]:
    return ["NOTE_OFF", str(event["note"])]

def control_change_event(event : Dict[Any, Any]) -> List[str]:
    return ["CONTROL_CHANGE", str(event['control']), str(event['value'])]

def set_tempo_event(event : Dict[Any, Any]) -> List[str]:
    return ["SET_TEMPO", str(event["tempo"])]

def time_signature_event(event : Dict[Any, Any]) -> List[str]:
    return ["TIME_SIG", str(event["numerator"]), str(event["denominator"])]

def time_shift_event(q_delta : float) -> List[str]:
    return ["TIME_SHIFT", str(q_delta)]

def resolution_ticks_event(resolution_ticks : float) -> List[str]:
    return ["RESOLUTION_TICKS", str(resolution_ticks)]

def detokenize_events(self, tokens: List[str]) -> MidiFile:
    """
    Rebuilds a MidiFile from a flat list of string tokens.
    """
    out = MidiFile()
    track = MidiTrack()
    out.tracks.append(track)

    abs_tick = 0
    i = 0
    resolution_ticks = 480  # Default fallback

    while i < len(tokens):
        token = tokens[i]

        if token == "RESOLUTION_TICKS":
            resolution_ticks = int(tokens[i+1])
            out.ticks_per_beat = resolution_ticks
            i += 2
            continue

        if token == "TIME_SHIFT":
            abs_tick += int(tokens[i+1])
            i += 2
            continue

        if token == "NOTE_ON":
            note = int(tokens[i+1])
            velocity = int(tokens[i+2])
            msg = Message('note_on', note=note, velocity=velocity, time=abs_tick)
            track.append(msg)
            abs_tick = 0
            i += 3
            continue

        if token == "NOTE_OFF":
            note = int(tokens[i+1])
            msg = Message('note_off', note=note, velocity=0, time=abs_tick)
            track.append(msg)
            abs_tick = 0
            i += 2
            continue

        if token == "CONTROL_CHANGE":
            control = int(tokens[i+1])
            value = int(tokens[i+2])
            msg = Message('control_change', control=control, value=value, time=abs_tick)
            track.append(msg)
            abs_tick = 0
            i += 3
            continue

        if token == "SET_TEMPO":
            tempo = int(tokens[i+1])
            msg = MetaMessage('set_tempo', tempo=tempo, time=abs_tick)
            track.append(msg)
            abs_tick = 0
            i += 2
            continue

        if token == "TIME_SIG":
            numerator = int(tokens[i+1])
            denominator = int(tokens[i+2])
            msg = MetaMessage('time_signature', numerator=numerator, denominator=denominator, time=abs_tick)
            track.append(msg)
            abs_tick = 0
            i += 3
            continue

        # Unknown token, skip

        # TODO: Handle cases, in which

        i += 1
    return out

class Tokenizer():
    #   Tokenizer class, that turns midifiles into str tokens and str tokens into midifiles
    #   doesnt write or load midifiles itself
    #   keeps track of map, that gets created in batches
    #   also saves processed dataset id, such that the maps can be saved with this id and correctly loaded with the training data later

    def __init__(self, processed_dataset_id : str):
        self.processed_dataset_id = processed_dataset_id

        self.type_map = {}
        self.pitch_map = {}
        self.duration_map = {}
        self.delta_offset_map = {}
        self.velocity_map = {}
        self.instrument_map = {}

    def _extend_maps(self, tokens):
        #   extends the maps of this tokenizer instance
        #
        #

        print("Start extending maps of tokens...")

        for event in tokens:
            if event.type not in self.type_map:
                self.type_map[event.type] = len(self.type_map)
            if event.pitch not in self.pitch_map:
                self.pitch_map[event.pitch] = len(self.pitch_map)
            if event.duration not in self.duration_map:
                self.duration_map[event.duration] = len(self.duration_map)
            if event.delta_offset not in self.delta_offset_map:
                self.delta_offset_map[event.delta_offset] = len(self.delta_offset_map)
            if event.velocity not in self.velocity_map:
                self.velocity_map[event.velocity] = len(self.velocity_map)
            if event.instrument not in self.instrument_map:
                self.instrument_map[event.instrument] = len(self.instrument_map)

        print("Finished extending maps of tokens.")


    def save_maps(self):            # preliminary
        #
        #
        #

        print("Start saving maps...")

        total_unique_tokens = len(self.type_map) + len(self.pitch_map)+len(self.duration_map)+len(self.delta_offset_map)+len(self.velocity_map)+len(self.instrument_map)

        print(f"Total unique tokens: {total_unique_tokens}")

        # save important information in tokenizer just in case, could also be saved in data
        self.sequence_length = SEQUENCE_LENGTH
        self.num_features_type = len(self.type_map)
        self.num_features_pitch = len(self.pitch_map)
        self.num_features_duration = len(self.duration_map)
        self.num_features_delta_offset = len(self.delta_offset_map)
        self.num_features_velocity = len(self.velocity_map)
        self.num_features_instrument = len(self.instrument_map)


        folder_path = os.path.join(TOKEN_MAPS_DIR, self.processed_dataset_id)
        os.makedirs(folder_path, exist_ok=False)
        with open(os.path.join(folder_path, "type_map.json"), "w") as f:
            json.dump(self.type_map, f, indent=4)
        with open(os.path.join(folder_path, "pitch_map.json"), "w") as f:
            json.dump(self.pitch_map, f, indent=4)
        with open(os.path.join(folder_path, "duration_map.json"), "w") as f:
            json.dump(self.duration_map, f, indent=4)
        with open(os.path.join(folder_path, "delta_offset_map.json"), "w") as f:
            json.dump(self.delta_offset_map, f, indent=4)
        with open(os.path.join(folder_path, "velocity_map.json"), "w") as f:
            json.dump(self.velocity_map, f, indent=4)
        with open(os.path.join(folder_path, "instrument_map.json"), "w") as f:
            json.dump(self.instrument_map, f, indent=4)

        print("Finished saving maps")


    def tokenize_events(self, midi: MidiFile) -> List[str]:
        events, ppq = _read_and_merge_events(midi)

        """
        Converts absolute‐tick events into tokens, quantizing time‐shifts:
        ('TIME_SHIFT', quantized_delta_ticks)
        ('NOTE_ON', (note, velocity))
        ('NOTE_OFF', note)
        ('CONTROL_CHANGE', (control, value))
        ('SET_TEMPO', tempo)
        ('TIME_SIG', (numerator, denominator))

        resolution_qn: grid in quarter‐notes (e.g. 0.25 = 16th notes)
        """
        tokens: List[str] = []
        last_tick = 0


        # number of ticks per quantization step
        resolution_ticks = max(1, round(ppq * QUANTIZATION_RESOLUTION))

        tokens.extend(resolution_ticks_event(ppq))

        for event in events:
            raw_delta = event['abs_tick'] - last_tick
            # quantize to nearest grid step
            q_delta = round(raw_delta / resolution_ticks) * resolution_ticks
            # advance our tick pointer by the quantized amount
            last_tick += q_delta

            if q_delta > 0:
                tokens.extend(time_shift_event(q_delta))

            t = event['type']
            if t == 'note_on':
                tokens.extend(note_on_event(event))
            elif t == 'note_off':
                tokens.extend(note_off_event(event))
            elif t == 'control_change':
                tokens.extend(control_change_event(event))
            elif t == 'set_tempo':
                tokens.extend(set_tempo_event(event))
            elif t == 'time_signature':
                tokens.extend(time_signature_event(event))
        return tokens
