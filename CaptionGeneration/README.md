# Image Captioning with CNN-RNN (ResNet50 + LSTM)

This project implements an image captioning system using a convolutional neural network (CNN) encoder (ResNet50) and a recurrent neural network (RNN) decoder (LSTM). The goal is to generate natural language descriptions (captions) for input images.
# Image Captioning - README

## 📁 Project Structure

```
project/
├── rollNumber_03_task1.py     # All code from data-loader to test & visualization
├── train.py                   # Training script
├── test.py                    # Testing Script
├── model.py                   # Captioning Model class
├── data_utils.py              # Dataset/dataloaders
├── weights/                   # Trained model weights  
│                              # (If weights are too large then upload them to your Drive and share the link in submission)
├── Report.pdf                 # Detailed report including analysis
├── graphs/                    # Contains all the graphs as mentioned
├── requirements.txt           # Dependencies
```

## 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training

To train the image captioning model:

```bash
python train.py
```

### Testing

To test the model:

```bash
python test.py
```

### Evaluate on Custom Images

To evaluate using a specific image path and display results:

```bash
python test_eval.py --image_path path/to/image.jpg
```

## 📝 Notes

* Ensure the image features are preprocessed as per the model's requirements.
* During testing, the best model is loaded automatically from the `weights/` directory.
* Use `rollNumber_03_task1.py` for full pipeline from data loading to visualization.
* Place all generated graphs in the `graphs/` folder.

## 📄 Report

Please refer to `Report.pdf` for a detailed explanation of the dataset, model architecture, training results, and analysis.

## 📊 Graphs

Training and validation loss curves, BLEU score trends, and other evaluation metrics should be stored in the `graphs/` folder.

## 🧾 Requirements

All dependencies required for the project are listed in `requirements.txt`. Install them before running the scripts.

---

