import os
from typing import List, Dict, Any, Tuple
from mido import MidiFile, MidiTrack, Message, MetaMessage

more_info = False

haydn = "haydn.midi"
chopin = "chopin.midi"
minuet = "minuet.mid"
toccata = "toccata.midi"
twinkle = "twinkle.mid"
kpop = "144.mid"



def read_and_merge_events(midi: MidiFile) -> Tuple[List[Dict], int]:
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



def quantize(value, resolution=1/32):
    return round(value / resolution) * resolution

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


def tokenize_events(midi: MidiFile, resolution_qn = 1/8) -> List[str]:
    events, ppq = read_and_merge_events(midi)

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
    resolution_ticks = max(1, round(ppq * resolution_qn))

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

def detokenize_events(tokens: List[str]) -> MidiFile:
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


###########################################################################################################################################

parse_dis = [haydn, chopin, minuet, toccata, twinkle]

for parse in parse_dis:
    file_path = f"data/midi/raw/Tokenize Dis/{parse}"

    midi = MidiFile(file_path)
    tokens = tokenize_events(midi)
    result = detokenize_events(tokens)

    u_input = input(f"Save result {parse} (y/n)?")

    if u_input == "y":
        file_name = os.path.basename(file_path)
        base_name = f"{os.path.splitext(file_name)[0]}_test"
        index = 0
        file_path = os.path.join("data/midi/results", f"{base_name}_{index}.mid")

        while os.path.exists(file_path):
            index += 1
            file_path = os.path.join("data/midi/results", f"{base_name}_{index}.mid")

        result.save(file_path)
        print("Saved result")
    elif u_input == "n":
        print("Wasnt saved")
