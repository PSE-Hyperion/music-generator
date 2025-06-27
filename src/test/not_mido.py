from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Dict, Tuple, Union
import os

# Set this to True for more detailed printouts
more_info = True

# ─── Step 1: Read & merge all tracks into absolute‐tick events (with pedals) ─────────
def read_and_merge_events(path: str) -> Tuple[List[Dict], int]:
    mid = MidiFile(path)
    ppq = mid.ticks_per_beat
    merged: List[Dict] = []
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
    merged.sort(key=lambda e: e['abs_tick'])
    return merged, ppq

# ─── Step 2: Tokenize into relative time + event tokens (including pedals), with quantization ─────────
def tokenize_events_custom(events: List[Dict], ppq: int, resolution_qn: float = 0.25) -> List[Tuple[str, ...]]:
    tokens: List[Tuple[str, ...]] = []
    last_tick = 0
    resolution_ticks = max(1, round(ppq * resolution_qn))
    for ev in events:
        raw_delta = ev['abs_tick'] - last_tick
        q_delta = round(raw_delta / resolution_ticks) * resolution_ticks
        last_tick += q_delta
        if q_delta > 0:
            tokens.append((f"DELTA_{q_delta}",))
        t = ev['type']
        if t == 'note_on':
            tokens.append((f"NOTE_{ev['note']}", f"VELOCITY_{ev['velocity']}"))
        elif t == 'note_off':
            tokens.append((f"NOTE_OFF_{ev['note']}",))
        elif t == 'control_change':
            tokens.append((f"CONTROL_{ev['control']}", f"VALUE_{ev['value']}"))
        elif t == 'set_tempo':
            tokens.append((f"TEMPO_{ev['tempo']}",))
        elif t == 'time_signature':
            tokens.append((f"TIME_SIG_{ev['numerator']}/{ev['denominator']}",))
    return tokens

# ─── Step 3: Build token maps ──────────────
def build_token_maps(tokens: List[Tuple[str, ...]]):
    token_types = {'NOTE': set(), 'VELOCITY': set(), 'DELTA': set(), 'CONTROL': set(), 'VALUE': set(), 'TEMPO': set(), 'TIME_SIG': set()}
    for token in tokens:
        for t in token:
            if t.startswith('NOTE_OFF_'):
                token_types['NOTE'].add(t)
            elif t.startswith('NOTE_'):
                token_types['NOTE'].add(t)
            elif t.startswith('VELOCITY_'):
                token_types['VELOCITY'].add(t)
            elif t.startswith('DELTA_'):
                token_types['DELTA'].add(t)
            elif t.startswith('CONTROL_'):
                token_types['CONTROL'].add(t)
            elif t.startswith('VALUE_'):
                token_types['VALUE'].add(t)
            elif t.startswith('TEMPO_'):
                token_types['TEMPO'].add(t)
            elif t.startswith('TIME_SIG_'):
                token_types['TIME_SIG'].add(t)
    maps = {k: {v: i for i, v in enumerate(sorted(token_types[k]))} for k in token_types}
    return maps

# ─── Step 4: Detokenize back into absolute‐tick event list ──────────────
def detokenize_events_custom(tokens: List[Tuple[str, ...]]) -> List[Dict]:
    events: List[Dict] = []
    abs_tick = 0
    for token in tokens:
        if token[0].startswith("DELTA_"):
            abs_tick += int(token[0].split("_")[1])
            continue
        elif token[0].startswith("NOTE_") and not token[0].startswith("NOTE_OFF_"):
            note = int(token[0].split("_")[1])
            velocity = int(token[1].split("_")[1]) if len(token) > 1 else 64
            events.append({'abs_tick': abs_tick, 'type': 'note_on', 'note': note, 'velocity': velocity})
        elif token[0].startswith("NOTE_OFF_"):
            note = int(token[0].split("_")[2])
            events.append({'abs_tick': abs_tick, 'type': 'note_off', 'note': note, 'velocity': 0})
        elif token[0].startswith("CONTROL_"):
            control = int(token[0].split("_")[1])
            value = int(token[1].split("_")[1]) if len(token) > 1 else 0
            events.append({'abs_tick': abs_tick, 'type': 'control_change', 'control': control, 'value': value})
        elif token[0].startswith("TEMPO_"):
            tempo = int(token[0].split("_")[1])
            events.append({'abs_tick': abs_tick, 'type': 'set_tempo', 'tempo': tempo})
        elif token[0].startswith("TIME_SIG_"):
            num, den = map(int, token[0].split("_")[2].split("/"))
            events.append({'abs_tick': abs_tick, 'type': 'time_signature', 'numerator': num, 'denominator': den})
    return events

# ─── Step 5: Write a single‐track MIDI from merged events ─────────────
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

if __name__ == '__main__':
    in_file = 'data/midi/raw/kpop_1_dataset/144.mid'
    print("\n" * 3)
    print(f"Reading MIDI file: {in_file}")
    events, ppq = read_and_merge_events(in_file)
    print(f"Read {len(events)} events (PPQ={ppq})")
    print("\nFirst 20 events:")
    for e in events[:20]:
        print(e)
    print("\n" * 2)

    resolution_qn = 0.125  # 16th notes
    tokens = tokenize_events_custom(events, ppq, resolution_qn)


    maps = build_token_maps(tokens)
    print("Token maps (sizes):")
    for k, v in maps.items():
        print(f"{k}: {len(v)}")
    if more_info:
        for k, v in maps.items():
            print(f"\n{k} map:")
            for token, idx in list(v.items())[:10]:
                print(f"  {token}: {idx}")
    print("\n" * 2)

    events_rt = detokenize_events_custom(tokens)


    u_input = input("Save result (y/n)? ")
    if u_input.lower() == "y":
        file_name = os.path.basename(in_file)
        base_name = f"{os.path.splitext(file_name)[0]}_custom"
        index = 0
        file_path = os.path.join("data/midi/results", f"{base_name}_{index}.mid")
        while os.path.exists(file_path):
            index += 1
            file_path = os.path.join("data/midi/results", f"{base_name}_{index}.mid")
        write_from_merged_events(events_rt, ppq, file_path)
        print("Saved result")
    else:
        print("Wasn't saved")
