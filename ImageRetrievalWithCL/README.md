# Contrastive Learning for Image Similarity

This project implements a **Siamese Network** for image similarity using contrastive learning. The model is trained to distinguish between similar and dissimilar image pairs.

## Project Structure
_(Provide a directory structure if needed)_

## Installation
Ensure you have Python 3.8+ installed.  
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model, run:
```bash
python train.py
```

### Testing
To evaluate the model on the test set:
```bash
python test.py
```

### Inference on Custom Images
To compare two images and check their similarity:
```bash
python test_eval.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

## Features
- Contrastive Loss for training a Siamese Network.
- Support for batch processing in validation/testing.
- Performance evaluation using metrics like accuracy, precision, recall.

## Results & Visualizations
### Loss & Accuracy Graphs
Tracked over training epochs to observe model performance.



## References
- Contrastive Learning in Computer Vision  
- Siamese Networks for One-shot Learning
