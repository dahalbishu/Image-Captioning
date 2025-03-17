import torch
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer
from dataloader import ImageCaption
from model import ViTGPT2Captioning
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--batch', type=int, default=8, help="Batch size")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--image_path', type=str, default=r"flicker8k\Images", help="Path to image folder")
parser.add_argument('--train_caption_file', type=str, default=r"flicker8k\captions.txt", help="Path to caption file")
parser.add_argument('--eval_caption_file', type=str, default=r"flicker8k\captions.txt", help="Path to caption file")

parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint file")
parser.add_argument('--lr', type=float, default=0.00005, help="Learning rate")
parser.add_argument('--eval', action='store_true', help="Run evaluation after training")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(image_dir, train_caption_file, eval_caption_file, num_epochs, batch_size, lr, resume_path=None):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ImageCaption(image_dir, train_caption_file, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if args.eval:
        eval_dataset = ImageCaption(image_dir, eval_caption_file, tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ViTGPT2Captioning().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # If a resume path is provided, load the checkpoint
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 0

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids[:, :-1], attention_mask[:, :-1])
            logits = outputs.logits

            loss = criterion(logits[:, :labels.size(1), :].reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}")

        os.makedirs("result", exist_ok=True)
        save_path = os.path.join("result", "checkpoint{}".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        
        
        # Run evaluation after each epoch if needed
        if args.eval:
            evaluate(model, eval_dataloader, tokenizer)



    
    

def evaluate(model, eval_dataloader, tokenizer):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images, input_ids[:, :-1], attention_mask[:, :-1])
            logits = outputs.logits

            loss = criterion(logits[:, :labels.size(1), :].reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(eval_dataloader):.4f}")


if __name__ == '__main__':

    train(args.image_path, args.train_caption_file, args.eval_caption_file, args.epochs, args.batch, args.lr, args.resume)
