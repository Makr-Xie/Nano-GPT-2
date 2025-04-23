# Nano-GPT-2 (124M) — From Scratch

This project is a faithful reproduction of the GPT-2 (124M) model following [Andrej Karpathy](https://www.youtube.com/@karpathy)'s video tutorial ["Let's build GPT: from scratch, in code, spelled out"](https://youtu.be/kCc8FmEb1nY).

The full process is
- Building the GPT-2 model architecture from scratch
- Optimizing training for speed and efficiency
- Following hyperparameters from the GPT-2/GPT-3 papers (I slightly modified them for better performance)
- Running a complete training cycle in 2xH100 GPUs for 4.5 hours

By the end, this implementation is about **90% similar to the original

---

## 🔧 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/Makr-Xie/Nano-GPT-2.git
cd Nano-GPT-2
```
### 2. Prepare Training Data

```bash
python fineweb.py
```
This will preprocess data into a format suitable for training.

### 3. Train the Model
Simple launch:
```bash
python train_gpt2.py
```
Multi-GPU (e.g. 2 GPUs):
```bash
torchrun --standalone --nproc_per_node=2 train_gpt2.py
```