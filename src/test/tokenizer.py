import os
from music21 import converter, stream, note, chord, instrument
from fractions import Fraction

more_info = False

original = "data/midi/raw/1bitdragon/chord_repeated_same.mid"

score = converter.parse(original)

flat = score.flatten().notesAndRests            # .stream() if we want to have more control or access


print("\n" * 5)

print("Reconstruction creates these events:")
for event in flat:
    print(f"{event}")

print("\n" * 5)


def quantize(value, resolution=1/32):
    return round(value / resolution) * resolution

def note_event(note : note.Note, instrument : instrument.Instrument, curr_offset : float, part_of_chord : bool = False, first_chord_note : bool = False):
    label = "NOTE_"
    if part_of_chord:
        label = "CHORD-NOTE_"
        if first_chord_note:
            label = "CHORD-NOTE-START_"             # was, wenn chords in unterschiedlichen parts gleichzeitig stattfinden

    return (
        f"{label}{note.pitch}",
        f"DURATION_{float(Fraction(note.quarterLength))}",
        f"OFFSET_{curr_offset}",
        f"VELOCITY_{note.volume.velocity}",
        f"INSTRUMENT_{instrument.instrumentName}"
    )

def rest_event(rest : note.Rest, curr_offset : float):
    return (
        "REST",
        f"DURATION_{float(Fraction(rest.quarterLength))}",
        f"OFFSET_{curr_offset}",
        "NO_VELOCITY",
        "NO_INSTRUMENT"
    )

embedded_tokens = []
curr_instrument = instrument.Instrument("Piano")

prev_offset = 0.0

note_counter = 0
rest_counter = 0
chord_counter = 0
note_in_chord_counter = 0

for event in flat:
        abs_offset = float(event.offset)
        curr_delta_offset = abs_offset - prev_offset
        prev_offset = abs_offset

        if isinstance(event, note.Note):
            embedded_tokens.append(note_event(event, curr_instrument, curr_delta_offset))
            note_counter += 1
        elif isinstance(event, chord.Chord):
            for i, n in enumerate(event.notes, start=0):
                if i == 0:
                    chord_delta_offset = curr_delta_offset
                    embedded_tokens.append(note_event(n, curr_instrument, curr_delta_offset, part_of_chord=True, first_chord_note=True))
                    note_in_chord_counter += 1
                    continue
                embedded_tokens.append(note_event(n, curr_instrument, chord_delta_offset, part_of_chord=True))
                note_in_chord_counter += 1
            chord_counter += 1
        elif isinstance(event, note.Rest):
            embedded_tokens.append(rest_event(event, curr_delta_offset))
            rest_counter += 1

# Optional print
if more_info:
    print(f"Events in embedded tokens: {len(embedded_tokens)}")
    print(f"Notes in embedded tokens: {note_counter}")
    print(f"Rests in embedded tokens: {rest_counter}")
    print(f"Chords in embedded tokens: {chord_counter}")
    print(f"Note in chords in embedded tokens: {note_in_chord_counter}")

###########################################################################################################################################



pitches = set()
durations = set()
offsets = set()
velocities = set()
instruments = set()

for embedded_token in embedded_tokens:          # maybe use embeddedtoken class
    pitch = embedded_token[0]
    pitches.add(pitch)
    duration = embedded_token[1]
    durations.add(duration)
    offset = embedded_token[2]
    offsets.add(offset)
    velocity = embedded_token[3]
    velocities.add(velocity)
    instr = embedded_token[4]
    instruments.add(instr)

pitch_map = {p: i for i, p in enumerate(sorted(pitches))}
duration_map   = {d: i for i, d in enumerate(sorted(durations))}
offset_map = {o: i for i, o in enumerate(sorted(offsets))}
velocity_map = {v: i for i, v in enumerate(sorted(velocities))}
instrument_map = {i: j for j, i in enumerate(sorted(instruments))}

# Optional print
if more_info:
    print(f"Pitch map size: {len(pitch_map)}")
    print(f"Dur map size: {len(duration_map)}")
    print(f"Offset map size: {len(offset_map)}")
    print(f"Veloc map size: {len(velocity_map)}")
    print(f"Instr map size: {len(instrument_map)}")

    print(f"Total unique tokens: {len(pitch_map)+len(duration_map)+len(offset_map)+len(velocity_map)+len(instrument_map)}")

###########################################################################################################################################

def is_embedded_token_rest(embedded_token) -> bool:
    return embedded_token[0] == "REST"

def is_embedded_token_note(embedded_token) -> bool:
    return embedded_token[0].startswith("NOTE")

def is_embedded_token_part_of_chord(embedded_token) -> bool:
    return embedded_token[0].startswith("CHORD-NOTE")

def is_first_chord_note(embedded_token) -> bool:
    return embedded_token[0].startswith("CHORD-NOTE-START")



print("\n" * 5)

print("Tokenization creates these tokens:")
for embedded_token in embedded_tokens:
    print(f"{embedded_token[0]}")

print("\n" * 5)




s = stream.Stream()
chord_notes = []
current_offset = 0.0  # absolute offset

note_in_chord_counter = 0

for embedded_token in embedded_tokens:
    if is_embedded_token_rest(embedded_token):
        dur = float(embedded_token[1].split("_")[1])
        delta_offset = float(embedded_token[2].split("_")[1])
        current_offset += delta_offset
        s.insert(current_offset, note.Rest(quarterLength=dur))

    elif is_embedded_token_note(embedded_token):
        pitch = embedded_token[0].split("_")[1]
        dur = float(embedded_token[1].split("_")[1])
        delta_offset = float(embedded_token[2].split("_")[1])
        velocity = int(embedded_token[3].split("_")[1])
        current_offset += delta_offset
        n = note.Note(pitch, quarterLength=dur)
        n.volume.velocity = velocity
        s.insert(current_offset, n)

    elif is_embedded_token_part_of_chord(embedded_token):
        # Flush previous chord if this is a new chord start
        if len(chord_notes) > 0 and is_first_chord_note(embedded_token):
            chord_offset = chord_notes[0][0]
            notes = [n for _, n in chord_notes]
            duration = notes[0].quarterLength
            s.insert(chord_offset, chord.Chord(notes, quarterLength=duration))
            note_in_chord_counter += len(notes)
            chord_notes = []

        token_type, pitch = embedded_token[0].split("_", 1)
        dur = float(embedded_token[1].split("_")[1])
        delta_offset = float(embedded_token[2].split("_")[1])
        velocity = int(embedded_token[3].split("_")[1])

        if len(chord_notes) == 0:
            current_offset += delta_offset  # Only update on first note of chord

        n = note.Note(pitch, quarterLength=dur)
        n.volume.velocity = velocity
        chord_notes.append((current_offset, n))

# Flush remaining chord notes
if chord_notes:
    chord_offset = chord_notes[0][0]
    notes = [n for _, n in chord_notes]
    duration = notes[0].quarterLength
    s.insert(chord_offset, chord.Chord(notes, quarterLength=duration))
    note_in_chord_counter += len(notes)


# Optional print
if more_info:
    print(f"Events in midi: {len(s) + note_in_chord_counter - len(s.recurse().getElementsByClass(chord.Chord))}")
    print(f"Notes in midi: {len(s.recurse().getElementsByClass(note.Note))}")
    print(f"Rests in midi: {len(s.recurse().getElementsByClass(note.Rest))}")
    print(f"Chords in midi: {len(s.recurse().getElementsByClass(chord.Chord))}")
    print(f"Note in chords in midi: {note_in_chord_counter}")


print("\n" * 5)

print("Reconstruction creates these events:")
for event in s:
    print(f"{event}")

print("\n" * 5)



###########################################################################################################################################

u_input = input("Save result (y/n)?")

if u_input == "y":
    file_name = os.path.basename(original)
    base_name = f"{os.path.splitext(file_name)[0]}_test"
    index = 0
    file_path = os.path.join("data/midi/results", f"{base_name}_{index}.mid")

    while os.path.exists(file_path):
        index += 1
        file_path = os.path.join("data/midi/results", f"{base_name}_{index}.mid")


    s.write("midi", fp=file_path)
    print("Saved result")
elif u_input == "n":
    print("Wasnt saved")
