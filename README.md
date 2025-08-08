# [WIP] DTTNet Music Source Separation model

## [NOTE] This model currently has issues with static noise in the output as well as unsatisfactory separation. Although it performs relatively well in training/validation loss, it is still a work in progress and will be completed later on.

# Navigation
- [Research Paper - Music Source Separation Based on a Lightweight Deep Learning Framework (DTTNET: DUAL-PATH TFC-TDF UNET)](#research-paper)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Training Performance](#training-performance)

# Research Paper - Music Source Separation Based on a Lightweight Deep Learning Framework (DTTNET: DUAL-PATH TFC-TDF UNET) <a id="research-paper"></a>

This model is a DTTNet deep-learning model based on the 2023 research paper [Music Source Separation Based on a Lightweight Deep Learning Framework (DTTNET: DUAL-PATH TFC-TDF UNET)](https://arxiv.org/abs/2309.08684), authored by Junyu Chen, Susmitha Vekkot, and Pancham Shukla. The model takes in a song's audio file and returns two audio files; one consists of the vocals, the other consists of the instrumentals, extracted from the original file. This architecture is trained using the [MUSDB18-HQ dataset](https://www.kaggle.com/datasets/quanglvitlm/musdb18-hq), credit to quanglvitlm on Kaggle.

<p align="center">
<img src="https://i.postimg.cc/L59QKQ24/x1.png" width="800">
</p>

# Requirements

```bash
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
datasets>=2.0.0
kagglehub>=0.2.0
```

- Python 3.8+
- CUDA-compatible GPU

  # Project Structure

```bash
mss/
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── best_model.pth             # pre-trained model weights
├── best_loss.json             # best validation loss tracker
├── data/
│   └── preprocessing.py
├── model/
│   ├── dttnet.py 
│   ├── tfc_tdf.py
│   ├── dual_path.py
├── utils/
│   ├── audio_utils.py
│   ├── loss.py
├── output/
│   ├── vocals.wav             # to be outputted
│   └── background.wav         # to be outputted
```
# Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/buibaogianguyen/music-source-seperation.git
   cd music-source-seperation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset (for training):**
   The training script automatically downloads the MUSDB18-HQ dataset via Kaggle Hub.

## Usage

### Training

To train the model from scratch:

```bash
python train.py
```

**Training Configuration:**
- **Learning Rate:** 0.0005
- **Epochs:** 100
- **Batch Size:** 10
- **Optimizer:** AdamW with weight decay (1e-5)
- **Scheduler:** CosineAnnealingWarmRestarts
- **Loss Function:** Time-Frequency Domain Loss (α=0.7)
- **Dataset:** MUSDB18-HQ

You may adjust these parameters depending on your setup.

### Inference

To separate an audio file into vocals and background, update the file path leading to the input audio file (.WAV) and run:

```bash
python inference.py
```

# Training Performance

The model's training performance proves to be solid. But, as mentioned, the model itself still struggles with static noise and 'half-complete' source separation, which will be worked on in the future.

<p align="center">
<img src="https://i.postimg.cc/9f2MhyRg/0-1094.png" width="800">
</p>

Full progress of the model's training

```bash
Epoch 1, Train Loss: 0.2891208291053772, Val Loss: 0.29857857525348663
Epoch 2, Train Loss: 0.24823167324066162, Val Loss: 0.23785166442394257
Epoch 3, Train Loss: 0.21603014320135117, Val Loss: 0.205967977643013
Epoch 4, Train Loss: 0.18552536964416505, Val Loss: 0.17759183794260025
Epoch 5, Train Loss: 0.17063419222831727, Val Loss: 0.16449007391929626
Saved new best model
Epoch 6, Train Loss: 0.15681180208921433, Val Loss: 0.1649671271443367
Epoch 7, Train Loss: 0.15007285624742508, Val Loss: 0.15963584929704666
Saved new best model
Epoch 8, Train Loss: 0.14800976365804672, Val Loss: 0.1505318433046341
Saved new best model
Epoch 9, Train Loss: 0.1447248011827469, Val Loss: 0.1493789330124855
Saved new best model
Epoch 10, Train Loss: 0.1397615984082222, Val Loss: 0.15211163461208344
Epoch 11, Train Loss: 0.1387902557849884, Val Loss: 0.1517241895198822
Epoch 12, Train Loss: 0.12761070802807808, Val Loss: 0.1447136327624321
Saved new best model
Epoch 13, Train Loss: 0.12443504333496094, Val Loss: 0.14417541772127151
Saved new best model
Epoch 14, Train Loss: 0.12363703101873398, Val Loss: 0.13459410518407822
Saved new best model
Epoch 15, Train Loss: 0.11078673824667931, Val Loss: 0.12862874567508698
Saved new best model
Epoch 16, Train Loss: 0.11849355399608612, Val Loss: 0.143567755818367
Epoch 17, Train Loss: 0.11161351650953293, Val Loss: 0.13464245200157166
Epoch 18, Train Loss: 0.11100345849990845, Val Loss: 0.1305430382490158
Epoch 19, Train Loss: 0.11026859357953071, Val Loss: 0.14883916825056076
Epoch 20, Train Loss: 0.11360457390546799, Val Loss: 0.11129475384950638
Saved new best model
Epoch 21, Train Loss: 0.10488172471523285, Val Loss: 0.1245957538485527
Epoch 22, Train Loss: 0.10293629318475724, Val Loss: 0.14757921546697617
Epoch 23, Train Loss: 0.11213375851511956, Val Loss: 0.13028812408447266
Epoch 24, Train Loss: 0.11036216989159583, Val Loss: 0.13250654190778732
Epoch 25, Train Loss: 0.10671921372413636, Val Loss: 0.13002275303006172
Epoch 26, Train Loss: 0.10597103908658027, Val Loss: 0.1474844589829445
Epoch 27, Train Loss: 0.111637032777071, Val Loss: 0.1396813839673996
Epoch 28, Train Loss: 0.10949349701404572, Val Loss: 0.14402896165847778
Epoch 29, Train Loss: 0.10342971086502076, Val Loss: 0.13571097701787949
Epoch 30, Train Loss: 0.1128155805170536, Val Loss: 0.10947781428694725
Saved new best model
```

# License

This project is licensed under the MIT License. See the `LICENSE` file for details (create one if needed).

# Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.
