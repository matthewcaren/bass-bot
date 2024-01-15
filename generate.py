from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import sys
import os
import midi

chunk_length = 4000 # how many midi events to generate
maxlen = 24 # how many events back the LSTM looks at


model = tf.keras.models.load_model("./model/model.h5")
model.load_weights("./model/weights.h5")

f = open('./miditext/source.txt', 'r')
music_as_chunks = []
for elm in f:
    music_as_chunks.append(elm.rstrip("\n"))
f.close()

resolution = 480

unique_chunks = sorted(list(set(music_as_chunks)))
chunk_indices = dict((c, i) for i, c in enumerate(unique_chunks))
indices_chunk = dict((i, c) for i, c in enumerate(unique_chunks))

start_index = random.randint(0, len(music_as_chunks) - maxlen - 1)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

sampled_chunks = random.sample(music_as_chunks, maxlen)
generated = []
part_of_song = music_as_chunks[0: maxlen]
generated.extend(part_of_song)
sys.stdout.write(str(generated))


for i in range(chunk_length):
    x = np.zeros((1, maxlen, len(unique_chunks)))
    for t, chunk in enumerate(part_of_song):
        x[0, t, chunk_indices[chunk]] = 1.

    preds = model.predict(x, verbose=0)[0]
    distributions = [0.2,  0.6, 1.0, 1.4]
    weight = [0.3, 0.5, 0.1, 0.1]
    distribution = np.random.choice(distributions, p=weight)
    next_index = sample(preds, distribution)
    if (i+maxlen) % 16 == 0:
        next_char = "45"
    else:
        next_char = indices_chunk[next_index]

    generated.extend([next_char])
    part_of_song = part_of_song[1:]
    part_of_song.extend([next_char])

    sys.stdout.flush()
print()

if not os.path.exists("./midi/"):
    os.mkdir("./midi/")
    os.mkdir("./midi/")
elif not os.path.exists("./midi/"):
    os.mkdir("./midi")
file = "./midi/generated-song.mid"

pattern = midi.Pattern(resolution=resolution)

track = midi.Track()
pattern.append(track)

time_since_last = 0

for chunk in generated:
    print(chunk)
    if chunk == "-":
        time_since_last += 120
    else:
        pitch = int(chunk)
        e = midi.NoteOnEvent(tick=time_since_last, channel=0, velocity=80, pitch=pitch)
        track.append(e)
        time_since_last = 120

end_event = midi.EndOfTrackEvent(tick=1)
track.append(end_event)

midi.write_midifile(file, pattern)
