# Automatic Chord Recognition with Transformers 🎸

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
visuals/                 # pictures of evaluation plots
splits_seed2025.npz      # Pre-saved train/val/test song splits
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

   * Python ≥ 3.9
   * PyTorch ≥ 2.0
   * `torch`, `torchvision`, `torchaudio`
   * `numpy`, `scipy`, `scikit-learn`
   * `matplotlib`, `seaborn`
   * `librosa`

4. Dataset

   We use GuitarSet, a collection of 360 annotated 30-second excerpts with hexaphonic guitar recordings.
   Annotations are provided in Harte notation and converted to frame-level labels aligned with Constant-Q Transform (CQT) spectrograms.
  
   (Optional) Download [GuitarSet dataset](https://github.com/marl/guitarset) and place the `.jams` and `.wav` files in a `data/` directory.

---

## 📊 Usage

All experiments are in the provided Jupyter notebook:

```bash
jupyter notebook Chord_Recognition.ipynb
```
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


## 📈 Results Showcase

Comparison of Baseline vs. Light Augmentation (SpecAugment) on GuitarSet (Maj/Min vocabulary).

| Setup         | Decoder  | Val Acc | Test Acc | Val WCSR | Test WCSR |
|---------------|----------|---------|----------|----------|-----------|
| **Baseline**  | Argmax   | 0.815   | 0.798    | 0.815    | 0.798     |
|               | Median   | 0.818   | 0.798    | 0.818    | 0.798     |
|               | Viterbi  | 0.824   | 0.798    | 0.824    | 0.798     |
| **Light Aug** | Argmax   | 0.814   | 0.809    | 0.814    | 0.809     |
|               | Median   | 0.814   | 0.809    | 0.814    | 0.809     |
|               | Viterbi  | 0.814   | 0.807    | 0.814    | 0.807     |

**Key Takeaways**:
- Light augmentation slightly boosts test accuracy (+1% absolute).  
- Median filtering and Viterbi smoothing help stabilize predictions but do not dramatically change WCSR.  
- The Transformer baseline is already strong and competitive with state-of-the-art results reported in MIR literature.

---

## 🎵 Pretrained Model

We provide the best-performing model checkpoint so you can reproduce our results without retraining from scratch.

### 📥 Download

The pretrained weights are **not stored directly in the repository** (due to GitHub’s file size limits).  
Instead, you can download them from the [Releases page](https://github.com/LidorGoldshmidt/chord-recognition-Project/releases)— look for the asset:

- `chord_tx_best.pt`

---

## 🔮 Future Work

* Expand vocabulary beyond maj/min to richer chord qualities (7th, sus, etc.)
* Handle class imbalance with re-weighting or focal loss
* Explore **multi-modal input** (e.g., MIDI, tablature)
* Test cross-dataset generalization (Isophonics, Beatles dataset)
* Move toward **real-time recognition** for interactive applications

---

## ⚡ Credits

* Dataset: [GuitarSet](https://github.com/marl/guitarset)
* Inspiration: Research in chord recognition with CNNs, HMMs, and Transformers
* Developed as part of an academic project on Automatic Music Transcription and Analysis.

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

