# Multilingual Text Classification with BERT

## Overview
This project implements a multilingual text classification system using BERT (Bidirectional Encoder Representations from Transformers). The model is trained on the XNLI (Cross-lingual Natural Language Inference) dataset and can classify text pairs in multiple languages as entailment, contradiction, or neutral.

## Features
- Utilizes the `bert-base-multilingual-cased` model
- Supports 15 languages including English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili, and Urdu
- Achieves approximately 70% accuracy on the validation set
- Includes a simple interface for making predictions on new text pairs

## Requirements
```
transformers==4.44.2
datasets==3.0.1
torch
scikit-learn
```

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/multilingual-text-classification.git
cd multilingual-text-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, run:
```python
python train.py
```

The training script will:
1. Load the XNLI dataset
2. Preprocess the data using the BERT tokenizer
3. Train the model for 3 epochs
4. Save the trained model and evaluation results

### Making Predictions
```python
from predict import predict

premise = "The man is eating pasta."
hypothesis = "The man is consuming food."
result = predict(premise, hypothesis)
print(f"Prediction: {result}")
```

## Model Performance
On the validation set, the model achieves:
- Accuracy: 70%
- F1 Score: 0.70
- Precision: 0.72
- Recall: 0.70

## Dataset
The XNLI dataset is used for training and evaluation. It contains:
- 392,702 training examples
- 2,490 validation examples
- 5,010 test examples

Each example consists of a premise, hypothesis, and label (entailment, contradiction, or neutral).

## Project Structure
```
├── train.py           # Script for training the model
├── predict.py         # Script for making predictions
├── requirements.txt   # Required packages
├── results/           # Saved model and training results
└── logs/             # Training logs
```

## Example Output
```
Premise: The man is eating pasta.
Hypothesis: The man is consuming food.
Prediction: entailment
```

## Limitations
- The model is trained on a subset of the full dataset for faster training
- Performance may vary across different languages
- The model may struggle with complex or nuanced relationships between texts

## Future Improvements
- Train on the full dataset
- Implement data augmentation techniques
- Add support for more languages
- Optimize hyperparameters for better performance
- Create a web interface for easy testing

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Hugging Face for providing the transformers library
- The XNLI dataset creators and contributors

## Citation
If you use this code in your research, please cite:
```
@misc{multilingual-text-classification,
  author = {Parvej Akhter},
  title = {Multilingual Text Classification with BERT},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/parvej8461/Bert_Model}
}
```
