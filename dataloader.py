import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2


class ImageCaption(Dataset):
    def __init__(self, image_folder, cap_file,tokenizer,max_length=50):
        self.image_folder = image_folder
        self.image_cap = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform =  transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.cap_file = cap_file

        # Reading captions from file
        with open(self.cap_file, 'r') as f:
            next(f)  # Skip the first line (column name)
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) < 2:
                    continue
                img_name, caption = parts
                self.image_cap.append((img_name,caption))


            

        
    def __len__(self):
        return len(self.image_cap)
    
    def __getitem__(self, idx):
        image_name,caption = self.image_cap[idx]
        image_path = os.path.join(self.image_folder, image_name)

        # Open image and convert it to RGB
        image = cv2.imread(image_path)  # Read image
        assert image is not None, f"Error in loading image {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transform(image)
        
        tokens = self.tokenizer(caption, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            "images": image,  # Image tensor
            "input_ids": tokens["input_ids"].squeeze(0),  # Tokenized caption
            "attention_mask": tokens["attention_mask"].squeeze(0),  # Mask for padding
            "labels": tokens["input_ids"].squeeze(0)  # Labels for training
        }













