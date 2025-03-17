FROM  python:3.10.16

WORKDIR /app

COPY requirements.txt /app/

# Install dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel  

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN apt-get update && apt-get install -y libgl1-mesa-glx 


RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN chmod +x train.py


CMD ["python", "train.py"]

