from torchvision.io import read_image
from torchvision.transforms import ToTensor
from pathlib import Path
import pathlib

class Portrait(Dataset):
    def __init__(self, img_dir : str, transform=None):
        self.paths = list(pathlib.Path(img_dir).glob("*/*.jpg"))
        self.transform = transform
    
    def load_image(self, idx: int) -> Image.Image:
        image_path = self.paths[idx]
        return Image.open(image_path)      

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx:int) -> torch.Tensor:
        image =  self.load_image(idx)
        if self.transform is not None:
            image = self.transform(image)

        return image
