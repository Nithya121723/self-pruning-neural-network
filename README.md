# self-pruning-neural-network

A PyTorch implementation where a neural network learns to prune its own 
weights during training — no post-training pruning step needed.

Built as part of the Tredence AI Engineering Internship case study.

## How to Run

```bash
pip install torch torchvision matplotlib
python self_pruning_network.py
```

CIFAR-10 downloads automatically on first run.

---

## Structure
```
├── self_pruning_network.py    
├── REPORT.md                  
└── gate_hist_lambda_*.png     
```
