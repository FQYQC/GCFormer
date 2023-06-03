import torch
import os
import json
from torch.utils.data import Dataset
import librosa
import numpy as np
import math

class GCDataset(Dataset):
    def __init__(self, path, maxlen=128):
        self.path = path
        self.data = []
        self.maxlen = maxlen
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            if filename.endswith(".json"):
                note, time = self.getfile(filename)
                self.data.append((note, time))


    def getfile(self, filename):
        with open(filename) as f:
            data = json.load(f)
        note = []
        time = []
        i = 1
        while True:
            try:
                notedata = data["note"+str(i)]
                if not (notedata["note"] == "" or notedata["rhythm"] == ""):
                    n = notedata["note"]
                    t = notedata["rhythm"]
                    t = -math.log2(float(t))
                    t = math.floor(t+0.5)+3
                    if t<1 or t>9:
                        print(notedata)
                        raise(
                            OverflowError
                        )
                    if notedata["note"] == "#":
                        n = 1 # rest
                    else:
                        try:
                            n = librosa.note_to_midi(n)
                        except:
                            raise(OverflowError)
                    note.append(n)
                    time.append(t)
                i += 1
            except KeyError:
                break
        assert len(note) == len(time)
        note.extend([0]*(self.maxlen-len(note)))
        time.extend([0]*(self.maxlen-len(time)))
        return torch.tensor(note), torch.tensor(time)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    dataset = GCDataset("Archive/train")
    print(dataset[0])
    print(len(dataset))



        
