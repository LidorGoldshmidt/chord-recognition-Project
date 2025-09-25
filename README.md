# 🎶 Transformer-Based Chord Recognition

This repository contains the code and experiments for our project on **automatic chord recognition** using a Transformer-based model trained on the **GuitarSet dataset**.
The system combines a convolutional front-end with a Transformer encoder to capture both spectral and temporal patterns in music audio.

---

## 🚀 Project Overview

* **Goal**: Automatically detect and classify chords from raw audio recordings.
* **Dataset**: [GuitarSet](https://github.com/marl/guitarset) – high-quality multi-annotator guitar recordings with chord labels in Harte notation.
* **Model**: CNN + Transformer Encoder
* **Evaluation**: Standard frame-level accuracy and **MIREX-style metrics** (Root, Maj/Min, Triads) using **Weighted Chord Symbol Recall (WCSR)**.
* **Key Result**: Competitive performance with state-of-the-art baselines (~81% WCSR in maj/min vocabulary).

---

## 📂 Repository Structure

```
Chord_Recognition.ipynb   # Main notebook with full pipeline
README.md                 # Project documentation
checkpoints/              # Saved model weights (best.pt)
splits_gs_seed42.npz      # Pre-saved train/val/test song splits
```

---

## 🔧 Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/chord-recognition.git
   cd chord-recognition
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Core dependencies include:

   * `torch`, `torchvision`, `torchaudio`
   * `numpy`, `scipy`, `scikit-learn`
   * `matplotlib`, `seaborn`
   * `librosa`

3. (Optional) Download [GuitarSet dataset](https://github.com/marl/guitarset) and place the `.jams` and `.wav` files in a `data/` directory.

---

## 📊 Usage

Run the main notebook step by step:

1. **Data Preparation**

   * Loads GuitarSet audio and annotations
   * Computes **Constant-Q Transform (CQT)**
   * Builds chord vocabulary (`majmin`, `no_bass`, or `full`)
   * Creates train/val/test splits at **song-level** to avoid leakage

2. **Training**

   * Transformer model with CNN front-end
   * Cross-entropy loss with label smoothing
   * Cosine learning rate schedule with warmup
   * Early stopping and checkpoint saving

3. **Evaluation**

   * Frame-level accuracy
   * Confusion matrix and class distribution
   * **MIREX WCSR** evaluation (root-only, maj/min, triads)
   * Post-processing with different decoders:

     * **Argmax** (baseline)
     * **Median filter** (temporal smoothing)
     * **Viterbi** (transition matrix consistency)

4. **Visualization**

   * Training/validation curves
   * Confusion matrices
   * Class distribution histograms
   * CQT spectrogram with chord label overlays

---

## 📈 Results 

| Setup     | Decoder | Val Acc | Test Acc | Test WCSR (Maj/Min) |
| --------- | ------- | ------- | -------- | ------------------- |
| Baseline  | Median  | 0.818   | 0.798    | 0.798               |
| Light Aug | Median  | 0.814   | 0.809    | 0.809               |

* Augmentation (SpecAugment) slightly improved **test accuracy** and WCSR.
* Viterbi decoding provided **temporal consistency** but limited gains compared to median filtering.

---

## 🔮 Future Work

* Expand vocabulary beyond maj/min to richer chord qualities (7th, sus, etc.)
* Handle class imbalance with re-weighting or focal loss
* Explore **multi-modal input** (e.g., MIDI, tablature)
* Test cross-dataset generalization (Isophonics, Beatles dataset)
* Move toward **real-time recognition** for interactive applications

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{chord-transformer2025,
  title={Transformer-Based Chord Recognition on GuitarSet},
  author={Lidor Goldshmidt and Ori Shinover},
  year={2025},
  note={GitHub repository}
}
```

---

⚡ **Acknowledgments**:

* GuitarSet dataset creators
* Previous chord recognition research (Li, Osmalskyj, Park et al.)
* HuggingFace / PyTorch community for model tools

