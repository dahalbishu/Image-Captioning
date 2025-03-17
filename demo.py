# import torch
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# from torchvision import transforms
# from transformers import GPT2Tokenizer
# from model import ViTGPT2Captioning
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', required=True, type=str, default=None, help="Path to checkpoint file")
# args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# def generate_caption(image_path, model_path):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     model = ViTGPT2Captioning().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
#     model.eval()
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         vision_outputs = model.vitmodel(image).last_hidden_state[:, 0, :]
#         vision_embeds = model.image_projection(vision_outputs).unsqueeze(1)
        
#         generated = torch.tensor([tokenizer.bos_token_id], device=device).unsqueeze(0)
#         for _ in range(50):
#             inputs_embeds = model.gpt2.get_input_embeddings()(generated)
#             inputs_embeds = torch.cat((vision_embeds, inputs_embeds), dim=1)
            
#             outputs = model.gpt2(inputs_embeds=inputs_embeds)
#             next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
#             if next_token.item() == tokenizer.eos_token_id:
#                 break
            
#             generated = torch.cat((generated, next_token), dim=1)
    
#     caption = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
#     return caption

# def upload_image():
#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
#     if not file_path:
#         return
    
#     img = Image.open(file_path)
#     img.thumbnail((300, 300))
#     img = ImageTk.PhotoImage(img)
#     image_label.config(image=img)
#     image_label.image = img
    
#     caption = generate_caption(file_path,args.checkpoint)
#     caption_label.config(text=f"Caption: {caption}")

# root = tk.Tk()
# root.title("Image Captioning with ViT-GPT2")
# root.geometry("400x500")

# btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
# btn_upload.pack(pady=10)

# image_label = tk.Label(root)
# image_label.pack()

# caption_label = tk.Label(root, text="Caption: ", wraplength=300, justify="center")
# caption_label.pack(pady=10)

# root.mainloop()


import torch
from torchvision import transforms
from transformers import GPT2Tokenizer
from model import ViTGPT2Captioning
import argparse
from PIL import Image

# Argument parsing for checkpoint and image path
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str, help="Path to checkpoint file")
parser.add_argument('--image_path', required=True, type=str, help="Path to image file")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate caption function
def generate_caption(image_path, model_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = ViTGPT2Captioning().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        vision_outputs = model.vitmodel(image).last_hidden_state[:, 0, :]
        vision_embeds = model.image_projection(vision_outputs).unsqueeze(1)
        
        generated = torch.tensor([tokenizer.bos_token_id], device=device).unsqueeze(0)
        for _ in range(50):
            inputs_embeds = model.gpt2.get_input_embeddings()(generated)
            inputs_embeds = torch.cat((vision_embeds, inputs_embeds), dim=1)
            
            outputs = model.gpt2(inputs_embeds=inputs_embeds)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated = torch.cat((generated, next_token), dim=1)
    
    caption = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return caption

# Main function to run the caption generation
def main():
    caption = generate_caption(args.image_path, args.checkpoint)
    print(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()
