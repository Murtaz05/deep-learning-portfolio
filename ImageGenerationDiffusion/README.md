# Image Generation with Diffusion Models

A DDPM-style diffusion model trained from scratch on a small animal-image dataset: learns to reverse a fixed noising process, so it can generate images by starting from pure noise and iteratively denoising.

## Approach

- **Forward process**: a linear beta schedule (`T=1000` steps) progressively adds Gaussian noise to training images.
- **Model**: a U-Net-style denoiser conditioned on the diffusion timestep via sinusoidal time embeddings, trained to predict the noise added at each step.
- **Reverse process**: starting from random noise `x_T`, the trained model iteratively denoises back to `x_0`.
- Trained for 100 epochs on 5 animal classes (Bear, Bird, Cat, Cow, Deer).

## Results

- Training loss decreased steadily over 100 epochs (`training_loss.png`).
- `intermediate_noising_steps.png` visualizes the forward process at `t = 200, 500, 800, 999` — showing gradual degradation from a clean image to near-pure noise.
- Successfully denoised from `x_999` (pure noise) back to a recognizable `x_0`, though outputs are still visibly noisy — with more training time/data the architecture has room to improve.

## Issues hit during training

- CUDA out-of-memory: fixed by reducing batch size and wrapping inference in `torch.no_grad()`.
- A device-side assert error from invalid label indices: fixed by restarting the kernel and validating label ranges before training.

## Dataset

A small multi-class animal image dataset (5 classes: Bear, Bird, Cat, Cow, Deer) — e.g. a subset of [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10). Not included in this repo. Place it here:

```
ImageGenerationDiffusion/
  dataset/
    animal_data/
      Bear/
      Bird/
      Cat/
      Cow/
      Deer/
```

`denoise_model.pth` (trained weights) is included in this repo.

## Setup

```bash
pip install torch torchvision pillow
python msds24040_05.py   # trains the model
```

Then open `test_single_sample.ipynb` to run inference: denoise a sample from pure noise and visualize the result, using the included `denoise_model.pth`.

## Project structure

```
model.py                    # noise schedule, time embedding, denoiser architecture
msds24040_05.py               # dataset loading + training script
main.ipynb                     # notebook version of the training pipeline
test_single_sample.ipynb        # inference: denoise one sample, visualize
denoise_model.pth                # trained weights (included)
training_loss.png                 # loss curve over 100 epochs
intermediate_noising_steps.png     # forward-process visualization at t=200/500/800/999
```

## Stack

PyTorch, torchvision.
