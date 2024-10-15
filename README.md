# MEIJU2025xyf

## How to use ```fusions_bimodal.py```

first, import each fusion:

```python
from fusions_bimodal import ConcatEarly, CrossAttention, TensorFusion, NLgate, MISA #import each fusion function
```

then, define a fusion_model. For example:

```python
fusion_model = ConcatEarly().to(device)
```

next, use the fusion_model:

```python
for epoch in range(num_epochs):
  model.train() #this is your main model
  fusion_model.train() #this is the fusion model

  for text, audio, vision, labels in train_loader:
      inputs = fusion_model(text.to(device), audio.to(device), vision.to(device)) #fuse the audio, text, and vision features after loading them
```
do not forget to do the same process in dev and test.

final, add the parameters into optimizer. For example:

```python
optimizer = torch.optim.AdamW(list(model.parameters()) + list(fusion_model.parameters()), lr=5e-4, eps=1e-8, weight_decay=1e-5)
```

Kindly cite

```
@article{li2024speech,
  title={Speech Emotion Recognition with ASR Transcripts: A Comprehensive Study on Word Error Rate and Fusion Techniques},
  author={Li, Yuanchao and Bell, Peter and Lai, Catherine},
  journal={arXiv preprint arXiv:2406.08353},
  year={2024}
}
```
