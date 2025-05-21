# Emotional Music Generation Project

## Project Overview
This project demonstrates a complete pipeline for generating emotional music using a Variational Autoencoder (VAE). It encompasses:
- Data Preprocessing
- Latent Representations Creation
- Model Training
- Music Generation

Each stage is organized into its respective Jupyter Notebook file.


We use the **[DEAM (Database for Emotional Analysis of Music)](http://cvml.unige.ch/databases/DEAM/)** dataset, which contains 1,802 excerpts of music annotated with both:

- **Static annotations**: average **valence** and **arousal** ratings for each song.
- **Dynamic annotations**: per-second emotional evolution (useful for chunk-wise labeling).

> **Citation**:
>
> Aljanaki, Anna, Y. Yang, and M. Soleymani. “Developing a benchmark for emotional analysis of music.” *PLOS ONE*, 2017.

You must manually download the dataset from the [official DEAM site](http://cvml.unige.ch/databases/DEAM/) and place the audio and annotation files into a folder such as `data/`.

## Environment Setup
1. Install Python 3.8+.
2. Create a new virtual environment:
   - On Windows:
     ```
     python -m venv env
     env\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     python3 -m venv env
     source env/bin/activate
     ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Execution Steps

### 1. Preprocessing (preprocessing.ipynb)
- Imports libraries and defines a DataPreparation class.
- Loads audio data and annotations.
- Analyzes the emotional distribution of the dataset.
- Processes and segments audio files into chunks (e.g., 5 and 10 seconds).
- Visualizes sample segments and analysis results.
- - Saves:
  - `.wav` audio chunks
  - `.json` metadata
  - `valence-arousal` visualizations

*Run this notebook first to generate processed data for downstream tasks.*

### 2. Latent Representations Creation (create_latent_representations.ipynb)
- Uses the preprocessed data.
- Leverages an EnCodec model to produce latent representations of audio segments.
- - Saves:
  - `.npy` latent arrays
  - `.csv` files with corresponding emotional labels
- Visualizes the latent space with sample plots.

*Execute this notebook after preprocessing to generate and inspect latent features.*

### 3. Model Training (train_model.ipynb)
- Configures training parameters (latent dimension, batch size, learning rate, etc.).
- Loads the latent representations dataset.
- Instantiates and trains the Emotional VAE model.
- - Tracks:
  - Total loss
  - Reconstruction (MSE)
  - KL Divergence
- Saves:
  - `.pt` model checkpoints
  - Reconstruction samples
  - Training curves
  - Latent space visualizations
- Visualizes training statistics such as total loss, reconstruction loss, and KL divergence.
- Generates sample outputs from the trained model.

*Run this notebook to train the VAE and obtain model checkpoints.*

### 4. Music Generation (generate_music.ipynb)
- Loads the trained VAE model from the checkpoint.
- Sets up sample generation with varying emotional conditions.
- Converts latent samples into audio signals with defined synthesis functions.
- - Samples new music segments conditioned on emotional values.
- Uses custom synthesis functions:
  - `create_musical_waveform()`
  - `create_melodic_music()`
  - `create_piano_melody()`
- Outputs:
  - `.wav` generated audio
  - Side-by-side waveform visualizations
- Saves generated audio and provides playback and visualization within the notebook.

*Execute this notebook last to generate and listen to emotional music based on the trained model.*

## 📁 Directory Structure

```
.
├── data/                           # Raw DEAM data (audio + annotations)
├── processed_data/                # Chunks and metadata
├── latent_representations/        # EnCodec latents and metadata
├── vae_model/                     # Trained model and visualizations
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── create_latent_representations.ipynb
│   ├── train_model.ipynb
│   └── generate_music.ipynb
├── requirements.txt
└── README.md
```
## Academic Context

This project was developed as part of a **master's thesis on AI-driven music generation**, combining state-of-the-art deep learning with affective computing principles. It offers a practical implementation of emotion-conditional generative modeling in audio.

## 🎧 Listen & Explore

Once training is complete, you can generate musical samples representing emotional states such as:

- Happy: `[1.0, 1.0]`
- Angry: `[0.0, 1.0]`
- Sad: `[0.0, 0.0]`
- Calm: `[1.0, 0.0]`

  
## Additional Notes
- Ensure all file paths within the notebooks match your local setup.
- Adjust configuration parameters (such as sample rates, output directories, and emotional conditions) as necessary.
- Comments and additional instructions in each notebook provide further guidance.
- Run the notebooks sequentially to ensure a consistent workflow and proper data flow.

## Conclusion
This repository provides a comprehensive framework for emotion-controlled music generation. From environment setup and preprocessing to training and audio synthesis, each step is detailed to facilitate a smooth implementation of the system.

Happy experimenting!
