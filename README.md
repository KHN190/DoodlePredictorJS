# Doodle Classifier (JS)

This repo is a fork of [the original](https://github.com/ROCCYK/DoodlePredictor/tree/main). It contains converted TensorFlow JS models.

This project uses a **MobileNet** architecture for classifying grayscale images. The model has been optimized to handle greyscale images efficiently, leveraging the depth and speed of MobileNet, which is typically used for lightweight image classification tasks. The project includes a data pipeline, model training, evaluation, and visualization of results, including a accuracy_loss graph to assess model performance.

## Baseline

```bash
python3 load_model.py
```

## JS Model

Graph model. Converted to directory [tfjs](./tfjs). Layered model conversion is not supported.
