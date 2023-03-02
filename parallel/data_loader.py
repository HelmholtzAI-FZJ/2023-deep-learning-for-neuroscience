import os
from io import BytesIO
import h5py 
import json
from PIL import Image

from torch.utils.data import Dataset

class ImagenetH5(Dataset):

    def __init__(self, root, h5_file, subset, transform=None):

        self.imgs = h5py.File(h5_file, 'r')[subset] 
    
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", subset)
        for entry in os.listdir(samples_dir):
            if subset == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    self.targets.append(target)
            elif subset == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                self.targets.append(target)

      
    def __len__(self) -> int:
        return self.imgs["images"].shape[0]

    def __getitem__(self, index: int):
        img_string = self.imgs["images"][index]

        with BytesIO(img_string) as byte_stream:
            img = Image.open(byte_stream)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
    
        return img, self.targets[index]

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

        self.samples = self.samples[:100000]
        self.targets = self.targets[:100000]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

