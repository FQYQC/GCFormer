import torch
import os
import json
from torch.utils.data import Dataset
import librosa
import math

class GCDataset(Dataset):
    def __init__(self, path, maxlen=128, diff = False):
        self.path = path
        self.data = []
        self.maxlen = maxlen
        self.diff = diff
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            if filename.endswith(".json"):
                d = self.getfile(filename)
                self.data.append(d)


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
                        raise(OverflowError)
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

        if self.diff:
            diffnote = []
            ref = note[0]
            for i in range(len(note)):
                if note[i] == 1:
                    diffnote.append(1)
                else:
                    if ref == 1:
                        ref = note[i]
                    diffnote.append(note[i]-ref+64)
       
        mask = [True]*len(note) + [False]*(self.maxlen-len(note))
        note.extend([0]*(self.maxlen-len(note)))
        time.extend([0]*(self.maxlen-len(time)))
        if self.diff:
            diffnote.extend([0]*(self.maxlen-len(diffnote)))
            return torch.tensor(note), torch.tensor(time), torch.tensor(mask), torch.tensor(diffnote)
        return torch.tensor(note), torch.tensor(time), torch.tensor(mask)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    dataset = GCDataset("Archive/train", 512, diff=True)
    print(dataset[0])
    print(len(dataset))



        
