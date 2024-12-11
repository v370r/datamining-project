import torch
from torch.nn.functional import softmax
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self,model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.max_tokens = self.tokenizer.model_max_length

        self.chunks = []# Current chunks
        self.sentiment_score = 0
        self.stride_length = 25 # Words from past chunk included in new chunk

    def add_text(self,text):
        tokens = self.tokenizer(text, truncation=False,return_tensors="pt")
        tokens_input = tokens["input_ids"][0]
        tokens_attention_mask = tokens["attention_mask"][0]

        new_chunks_input = [tokens_input[i:i + self.max_tokens] for i in range(0, len(tokens_input), self.max_tokens - self.stride_length)]
        new_chunks_attention = [tokens_attention_mask[i:i + self.max_tokens] for i in range(0, len(tokens_attention_mask), self.max_tokens - self.stride_length)]

        labeled_chunks = [{"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}
                  for input_ids, attention_mask in zip(new_chunks_input, new_chunks_attention)]
        
        self.chunks.extend(labeled_chunks)

    def reset(self):
        self.chunks = []
        self.sentiment_score = []

    def get_sentiment(self):
        """
            Returns sentiment of all the added text so far. Does not reset after use! use .reset() to do so or if you want 
        """
        if(len(self.chunks) == 0):
            return
        retarr = []
        for chunk in self.chunks:
            with torch.no_grad():
                logits = self.model(**chunk).logits
            # Subtract or add probabliity of positive or negative
            retarr.append(logits.argmax())
        return np.array(retarr,dtype=np.float32).mean()

sa = SentimentAnalyzer()

# txt2 = """
#  AI and rising geopolitical tensions, the USG has changed and may again change the export control rules at any time and further subject a wider range of our products to export restrictions and licensing requirements, negatively impacting our business and financial results. In the event of such change, we may be unable to sell our inventory of such products and may be unable to develop replacement products not subject to the licensing requirements, effectively excluding us from all or part of the China market, as well as other impacted markets, including the Middle East.
# """
# sa.add_text(txt2)

# with open("bedbath.txt","r") as f:
#     sa.add_text(f.read())

# print(sa.get_sentiment())

# with open("value_2024-12-03_145111.txt","r") as f:
#     txt = f.read().split(".")
#     findings = []
#     for sentence in txt:
        
#         logits = sa.model(**sa.tokenizer(sentence,truncation=True,return_tensors="pt")).logits
#         findings.append(logits.argmax())
#         if(logits.argmax() == 0):
#             print("**"*10)
#             print(sentence)
#     print(np.array(findings,dtype=np.float32).mean())
