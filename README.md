# MEIJU2025xyf

## How to use ```fusions_bimodal.py```

first, import each fusion:

```python
from fusions_bimodal import ConcatEarly, CrossAttention, TensorFusion, NLgate, MISA, ModalityGatedFusion #import each fusion function, except late fusion
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

  for text, audio, labels in train_loader:
      inputs = fusion_model(text.to(device), audio.to(device)) #fuse the audio and text features after loading them
```
do not forget to do the same process in dev and test.

final, add the parameters into optimizer. For example:

```python
optimizer = torch.optim.AdamW(list(model.parameters()) + list(fusion_model.parameters()), lr=5e-4, eps=1e-8, weight_decay=1e-5)
```

Kindly cite

```
@inproceedings{li2024fusion,
  title={Speech Emotion Recognition with ASR Transcripts: A Comprehensive Study on Word Error Rate and Fusion Techniques},
  author={Li, Yuanchao and Bell, Peter and Lai, Catherine},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
  organization={IEEE}
}
```
