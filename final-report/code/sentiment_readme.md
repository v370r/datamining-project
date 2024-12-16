# Sentiment Analysis Using DistilBERT

This repository implements a sentiment analysis tool using a fine-tuned DistilBERT model from Hugging Face. The tool processes textual data in chunks, performs sentiment analysis, and returns the average sentiment score.

---

## Features

- **Pretrained Model**: Utilizes the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face.
- **Chunk-Based Processing**: Splits large text into manageable chunks, incorporating a stride mechanism to maintain context.
- **Batch Processing**: Processes multiple chunks in batches for faster computation.
- **Flexible Device Support**: Automatically selects the best available device (CPU, GPU, or Apple MPS).

---

## Requirements

- Python 3.8+
- Libraries:
  - torch
  - transformers
  - numpy

Install the required libraries using pip:
```bash
pip install torch transformers numpy
```

---

## Code Overview

### Key Components

#### 1. **Initialization**
The `SentimentAnalyzer` class sets up the tokenizer and model, selecting the optimal computation device:
```python
class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = ("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else
                       "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.max_tokens = self.tokenizer.model_max_length
        self.chunks = []
        self.sentiment_score = 0
        self.stride_length = 25
```

#### 2. **Adding Text**
Splits input text into overlapping chunks to handle sequences longer than the model’s maximum token limit:
```python
def add_text(self, text):
    tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
    tokens_input = tokens["input_ids"][0]
    tokens_attention_mask = tokens["attention_mask"][0]

    new_chunks_input = [tokens_input[i:i + self.max_tokens]
                         for i in range(0, len(tokens_input), self.max_tokens - self.stride_length)]

    new_chunks_attention = [tokens_attention_mask[i:i + self.max_tokens]
                             for i in range(0, len(tokens_attention_mask), self.max_tokens - self.stride_length)]

    self.chunks.extend([
        {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}
        for input_ids, attention_mask in zip(new_chunks_input, new_chunks_attention)
    ])
```

#### 3. **Calculating Sentiment**
Computes the average sentiment score for all added text:
```python
def get_sentiment_batch(self, batch_size=32):
    full_chunks, partial = [], []

    for chunk in self.chunks:
        if len(chunk["input_ids"][0]) == 512:
            full_chunks.append(chunk)
        else:
            partial.append(chunk)

    if len(full_chunks) > 0:
        batched_input = {key: torch.cat([chunk[key] for chunk in full_chunks], dim=0) for key in full_chunks[0].keys()}

    partial_logits = []
    with torch.no_grad():
        if len(full_chunks) > 0:
            logits = self.model(**batched_input).logits
        if len(partial) > 0:
            for p in partial:
                partial_logits.append(self.model(**p).logits)

    retarr = []
    if len(full_chunks) > 0:
        retarr.extend(logits.argmax(dim=1).tolist())
    if len(partial) > 0:
        for pl in partial_logits:
            retarr.extend(pl.argmax(dim=1).tolist())

    return np.array(retarr, dtype=np.float32).mean() if len(retarr) > 0 else 0
```

#### 4. **Resetting State**
Clears all stored chunks and resets the sentiment score:
```python
def reset(self):
    self.chunks = []
    self.sentiment_score = []
```

---

## Usage

### Example

```python
from sentiment_analyzer import SentimentAnalyzer

sa = SentimentAnalyzer()

# Load text from a file
with open("cleaned_text.txt", "r") as f:
    text = f.read()

# Add text to the analyzer
sa.add_text(text)

# Get sentiment score
sentiment_score = sa.get_sentiment_batch()
print(f"Sentiment Score: {sentiment_score}")

# Reset for new input
sa.reset()
```

---

## Advantages

- Efficient processing of large texts through chunking and batching.
- Leveraging a state-of-the-art pretrained model for sentiment analysis.
- Device-agnostic implementation for flexible deployment.

---

## Future Enhancements

- Improve chunking by adding context overlap across sentences.
- Extend functionality to include multi-class sentiment analysis.
- Optimize for faster inference on large datasets.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the pretrained models and tokenizer.
- Contributors to open-source libraries used in this project.

