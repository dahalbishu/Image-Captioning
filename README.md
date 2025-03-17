# Image Captioning

This project implements an image captioning model using the Flickr8K dataset. It utilizes a Vision Transformer model and a GPT-2  model for generating captions based on the image content. 

## Dataset Structure

The dataset is organized as follows:

```
root/
|
|-- dataset/
|---- |
|---- |-- Images/
|---- |---- |-- 0.jpg
|---- |---- |-- 1.jpg
|---- |---- |-- ...
|---- |
|---- |-- caption_train.txt
|---- |-- caption_eval.txt (optional)
```

- `Images/`: Contains all the images in the dataset (e.g., `0.jpg`, `1.jpg`, etc.).
- `caption_train.txt`: Contains the training captions in the format `image_name.jpg, caption text`.
- `caption_eval.txt`: (Optional) Contains the evaluation captions in the same format.

## Dataset Format

It is optional to divide `caption.txt` of the Flickr dataset into train and eval text files.

Each line in the `caption_train.txt` and `caption_eval.txt` files should follow this format:

```
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg,A girl going into a wooden building .
1000268201_693b08cb0e.jpg,A little girl climbing into a wooden playhouse .
```

## Setup

Before running the scripts, make sure you have the required dependencies installed. You can use the following commands to set up the environment:

```bash
# Create and activate conda environment
conda create --name imagecaption python=3.10.16 -y
conda activate imagecaption

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install additional dependencies
pip install -r requirements.txt
```

## Training the Model

To train the model, run the following command:

```bash
python train.py --batch=8 --epochs=20 --image_path <path to image folder> --train_caption_file <path to train caption txt file> --eval_caption_file <path to eval txt file> --lr 0.00005 --eval 
```

### Parameters:
- `--batch`: Number of samples per batch (default: 8).
- `--epochs`: Number of training epochs (default: 20).
- `--image_path`: Path to the folder containing training images.
- `--train_caption_file`: Path to the training caption text file.
- `--eval_caption_file`: Path to the evaluation caption text file.
- `--lr`: Learning rate for the optimizer (default: 0.00005).
- `--eval`: Flag indicating whether to evaluate the model at each epoch.
- `--resume <path_to_checkpoint>`: Path to the checkpoint file to resume training.

### Checkpoint Saving:
- During training, checkpoints will be saved in the `result` directory.

## Running the Demo

After training, you can run the demo script to perform inference on a new image. Use the following command:

```bash
python demo.py --checkpoint <path to checkpoint> --image_path <path to image>
```

### Parameters:
- `--checkpoint`: Path to the saved checkpoint file of the trained model.
- `--image_path`: Path to the image you want to test with.

## Building the Docker Image

To build the Docker image, run the following command:
```sh
docker build -t imagecap:1.0 .
```

## Training the Model using Docker

```sh
docker run --gpus all \
    -v "<path to dataset folder>:/app/flicker8k" \
    -v "<path to checkpoint folder>:/app/checkpoints" \
    imagecap:1.0 \
    python train.py \
    --train_caption_file="/app/flicker8k/captions_train.txt" \
    --image_path="/app/flicker8k/Images"
```

Other parameters can be specified as needed.

## Running a Demo using Docker

```sh
docker run --rm --gpus all \
    -v "<path to image>:/app/projects/test.jpg" \
    -v "<path to checkpoint>:/app/result/demo_checkpoint" \
    imagecap:1.0 \
    python demo.py \
    --checkpoint=/app/result/demo_checkpoint \
    --image_path="/app/projects/test.jpg"
```

Modify `<path to dataset folder>`, `<path to checkpoint folder>`, `<path to image>`, and `<path to checkpoint>` as required.
