import os
import midi

pattern = midi.read_midifile("source.mid")
chunk_str_list = []

elapsed_time = 0
elapsed_sixteenths = 0
elapsed_sixteenths_prev = 0

for i, chunk in enumerate(pattern[0]):

    chunk_str = ""
    elapsed_time += chunk.tick

    elapsed_sixteenths = int(elapsed_time/120)


    if (chunk.name == "Note On"):
        chunk_str = str(chunk.pitch)

        for i in range(elapsed_sixteenths - elapsed_sixteenths_prev):
            chunk_str_list.append("-")

        chunk_str_list.append(chunk_str)

    elapsed_sixteenths_prev = elapsed_sixteenths


if not os.path.exists("./miditext/"):
    os.mkdir("./miditext/")
    os.mkdir("./miditext/")
elif not os.path.exists("./miditext/"):
    os.mkdir("./miditext/")

f = open('./miditext/source.txt', 'w')
for elm in chunk_str_list:
    f.write(str(elm) + "\n")
f.close()
