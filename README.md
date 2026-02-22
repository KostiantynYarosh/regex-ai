# Regex AI
Text to regex generation with visual explanations.
![Image](https://github.com/user-attachments/assets/fe424b1f-6176-4ae3-b87a-a936dc03737e)
**Tech Stack**
- **Frontend:** React, Vite, Tailwind v4
- **Backend:** Python FastAPI
- **Model:** Fine-tuned `google-t5/t5-base` (PyTorch, HuggingFace Transformers)

**Run Backend**
```bash
cd model
python serve.py
```

**Run Frontend**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

**Model Training**

**Dataset**
Training data is built from multiple sources (i-hate-regex, KB13, NL-RX-Synth, regex101). Raw pairs go through filtering pipeline that removes invalid regex, non-standard operators, and low-quality descriptions.
```bash
python scripts/build_dataset.py
```

**Supervised Training**
T5-base fine-tuned on (description, regex) pairs.
- **Cosine LR schedule** - learning rate warms up from 0 to max, then smoothly decays following a cosine curve for aggressive early learning and fine-grained tuning at the end.
- **Gradient accumulation** - accumulates gradients over N batches before updating weights, simulating larger batch size without extra VRAM.

```bash
python scripts/train.py --model-name google-t5/t5-base --epochs 5 --batch-size 16 --grad-accum 8
```

**Evaluation Metrics**
- **Val Loss** - cross-entropy on validation set
- **Compile Rate** - % of generated regex that are syntactically valid
- **Exact Match** - % identical to reference regex
- **Val Score** - blended metric combining string