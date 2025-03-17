
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2Tokenizer, GPT2LMHeadModel

class ViTGPT2Captioning(nn.Module):
    def __init__(self, vit_model='google/vit-base-patch16-224', gpt2_model='gpt2'):
        super(ViTGPT2Captioning, self).__init__()
        self.vitmodel = ViTModel.from_pretrained(vit_model)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        
        # Ensure tokenizer has padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_projection = nn.Linear(768,self.gpt2.config.hidden_size)

        self.gpt2.resize_token_embeddings(len(self.tokenizer))



    def forward(self, images, input_ids, attention_mask=None, labels=None):
        vit_output = self.vitmodel(images).last_hidden_state[:, 0, :]
        visual_embedding = self.image_projection(vit_output).unsqueeze(1)

        # Get input embeddings from GPT-2 tokenizer's word embeddings
        input_embeds = self.gpt2.transformer.wte(input_ids)
        
        # Concatenate the visual embedding with the input embeddings
        input_embeds = torch.cat([visual_embedding, input_embeds], dim=1)

        # Ensure the attention_mask is updated accordingly
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1, device=attention_mask.device), attention_mask], dim=1)

        # Ensure labels are shifted properly for causal language modeling
        if labels is not None:
            labels = torch.cat([torch.full((labels.shape[0], 1), -100, device=labels.device), labels], dim=1)  # Use -100 to ignore padding tokens

        # Forward pass through GPT-2
        outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return outputs


