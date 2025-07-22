from typing import Final

from mido import MidiFile, MidiTrack

MAX_MIDI_NOTE: Final[int] = 127


def transpose(midi_file: MidiFile, semitone_shift: int) -> MidiFile:
    """
    Creates a new copy of given midi file, where all notes are transposed by the given semitone shift.

    It does this by iterating over every event in the original midi file, copying all entries to a new
    midi file, except note events (note on and note off), which are transposed and than added to the
    new midi file.
    """

    transposed = MidiFile()
    transposed.ticks_per_beat = midi_file.ticks_per_beat

    for track in midi_file.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                new_msg = msg.copy(note=msg.note + semitone_shift)
                new_track.append(new_msg)
            else:
                new_track.append(msg.copy())
        transposed.tracks.append(new_track)

    return transposed


def read_and_merge_events(midi: MidiFile) -> tuple[list[dict], int]:
    """
    Merge tempo, time signature, note, and pedal (control change 64,66,67) events
    from all tracks into a global, time sorted list with absolute ticks.

    Returns a list of dictionaries (key: event type, value: event value) sorted by absolute ticks
    and the ticks per beat (ppq)

    Dev: This is an unchanged version of this method from the first mido implementation branch.
    """

    ppq = midi.ticks_per_beat
    merged: list[dict] = []

    # Meta events (tempo, time_signature) from track 0
    abs_tick = 0
    for msg in midi.tracks[0]:
        abs_tick += msg.time
        if msg.is_meta and msg.type in ("set_tempo", "time_signature"):
            entry = {"abs_tick": abs_tick, "type": msg.type}
            if msg.type == "set_tempo":
                entry["tempo"] = msg.tempo
            else:
                entry["numerator"] = msg.numerator
                entry["denominator"] = msg.denominator
            merged.append(entry)

    # Note and pedal events from every track
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            # Pedal CC: 64 = sustain, 66 = sostenuto, 67 = soft
            if msg.type == "control_change" and msg.control in (64, 66, 67):
                merged.append(
                    {"abs_tick": abs_tick, "type": "control_change", "control": msg.control, "value": msg.value}
                )
            # Note on
            elif msg.type == "note_on" and msg.velocity > 0:
                merged.append({"abs_tick": abs_tick, "type": "note_on", "note": msg.note, "velocity": msg.velocity})
            # Note off (or on with zero velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                merged.append({"abs_tick": abs_tick, "type": "note_off", "note": msg.note, "velocity": 0})

    # Sort globally by absolute tick
    merged.sort(key=lambda e: e["abs_tick"])
    return merged, ppq
