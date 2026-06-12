import os
from PIL import Image

class Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        for label, sub in [('real', 'real'), ('fake', 'fake')]:
            folder = os.path.join(root_dir, sub)
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith(('.png','.jpg','.jpeg','.webp')):
                        self.samples.append((os.path.join(folder, fname), 0 if label=='real' else 1))
        # Also support per-generator subfolders inside fake/
        fake_root = os.path.join(root_dir, 'fake')
        if os.path.isdir(fake_root):
            for gen_dir in os.listdir(fake_root):
                gen_path = os.path.join(fake_root, gen_dir)
                if os.path.isdir(gen_path):
                    for fname in os.listdir(gen_path):
                        if fname.lower().endswith(('.png','.jpg','.jpeg','.webp')):
                            self.samples.append((os.path.join(gen_path, fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return Image.open(path).convert('RGB'), label
