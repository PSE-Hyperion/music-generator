from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Dict, Tuple, Union

# ─── Step 1: Read & merge all tracks into absolute‐tick events (with pedals) ─────────
def read_and_merge_events(path: str) -> Tuple[List[Dict], int]:
    """
    Merge tempo, time‐signature, note, and pedal (control change 64,66,67) events
    from all tracks into a global, time‐sorted list with absolute ticks.
    """
    mid = MidiFile(path)
    ppq = mid.ticks_per_beat
    merged: List[Dict] = []

    # Meta events (tempo, time_signature) from track 0
    abs_tick = 0
    for msg in mid.tracks[0]:
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
    for track in mid.tracks:
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


# ─── Step 2: Tokenize into relative time + event tokens (including pedals), with quantization ─────────
Token = Tuple[str, Union[int, Tuple[int, int]]]

def tokenize_events(
    events: List[Dict],
    ppq: int,
    resolution_qn: float = 0.25
) -> List[Token]:
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
    tokens: List[Token] = []
    last_tick = 0

    # number of ticks per quantization step
    resolution_ticks = max(1, round(ppq * resolution_qn))

    for ev in events:
        raw_delta = ev['abs_tick'] - last_tick
        # quantize to nearest grid step
        q_delta = round(raw_delta / resolution_ticks) * resolution_ticks
        # advance our tick pointer by the quantized amount
        last_tick += q_delta

        if q_delta > 0:
            tokens.append(('TIME_SHIFT', q_delta))

        t = ev['type']
        if t == 'note_on':
            tokens.append(('NOTE_ON', (ev['note'], ev['velocity'])))
        elif t == 'note_off':
            tokens.append(('NOTE_OFF', ev['note']))
        elif t == 'control_change':
            tokens.append(('CONTROL_CHANGE', (ev['control'], ev['value'])))
        elif t == 'set_tempo':
            tokens.append(('SET_TEMPO', ev['tempo']))
        elif t == 'time_signature':
            tokens.append(('TIME_SIG', (ev['numerator'], ev['denominator'])))
    return tokens


# ─── Step 3: Detokenize back into absolute‐tick event list ──────────────
def detokenize_events(tokens: List[Token]) -> List[Dict]:
    """
    Rebuilds the merged event list from tokens, computing absolute ticks.
    """
    events: List[Dict] = []
    abs_tick = 0

    for kind, val in tokens:
        if kind == 'TIME_SHIFT':
            abs_tick += val
            continue

        entry: Dict = {'abs_tick': abs_tick}
        if kind == 'NOTE_ON':
            note_, vel = val  # type: ignore
            entry.update(type='note_on', note=note_, velocity=vel)
        elif kind == 'NOTE_OFF':
            entry.update(type='note_off', note=val, velocity=0)
        elif kind == 'CONTROL_CHANGE':
            control, value = val  # type: ignore
            entry.update(type='control_change', control=control, value=value)
        elif kind == 'SET_TEMPO':
            entry.update(type='set_tempo', tempo=val)  # type: ignore
        elif kind == 'TIME_SIG':
            num, den = val  # type: ignore
            entry.update(type='time_signature', numerator=num, denominator=den)
        else:
            continue

        events.append(entry)

    return events


# ─── Step 4: Write a single‐track MIDI from merged events ─────────────
def write_from_merged_events(events: List[Dict], ppq: int, out_path: str):
    out = MidiFile()
    out.ticks_per_beat = ppq
    track = MidiTrack()
    out.tracks.append(track)

    last_tick = 0
    for ev in events:
        delta = ev['abs_tick'] - last_tick
        last_tick = ev['abs_tick']

        t = ev['type']
        if t == 'note_on':
            msg = Message('note_on', note=ev['note'], velocity=ev['velocity'], time=delta)
        elif t == 'note_off':
            msg = Message('note_off', note=ev['note'], velocity=0, time=delta)
        elif t == 'control_change':
            msg = Message('control_change', control=ev['control'], value=ev['value'], time=delta)
        elif t == 'set_tempo':
            msg = MetaMessage('set_tempo', tempo=ev['tempo'], time=delta)
        else:  # time_signature
            msg = MetaMessage('time_signature',
                              numerator=ev['numerator'],
                              denominator=ev['denominator'],
                              time=delta)
        track.append(msg)

    out.save(out_path)
    print(f"Wrote reconstructed MIDI (with pedals) to {out_path} (PPQ={ppq})")


# ─── Full round‐trip in __main__ ──────────────────────────────────────
if __name__ == '__main__':
    file_name = "chopin"
    in_file  = f"data/midi/raw/Tokenize Dis/{file_name}.midi"
    events, ppq = read_and_merge_events(in_file)

    # Quantization grid: change this to 0.25, 0.125, 0.0625 etc.
    resolution_high = 0.0625
    resolution_mid = 0.125
    resolution_low = 0.25
    resolution_qn = resolution_high

    # Tokenize with quantization
    tokens = tokenize_events(events, ppq, resolution_qn)
    print(f"Generated {len(tokens)} tokens (quantized to {resolution_qn} qn)")

    # optional json for unique tokens
    if True:
        appended_tokens = []
        for token in tokens:
            appended_token = ""
            for feature in token:
                appended_token += f"_{str(feature)}"
            appended_tokens.append(appended_token)

        unique_tokens = sorted(set(appended_tokens))
        token_to_int = {token: idx for idx, token in enumerate(unique_tokens)}

        import json
        with open("data/processed/unique_mido_events.json", "w") as f:
            json.dump(token_to_int, f, indent=4)

    # (Feed `tokens` into your AI; produce `tokens_out`)
    tokens_out = tokens  # for test round‐trip

    # Detokenize
    events_rt = detokenize_events(tokens_out)

    if False:
        # Write back to MIDI
        out_file = f"data/midi/results/mido_{file_name}.midi"
        write_from_merged_events(events_rt, ppq, out_file)
