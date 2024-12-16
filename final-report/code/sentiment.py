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

            retarr.append(logits.argmax())
        return np.array(retarr,dtype=np.float32).mean()
    
    def get_sentiment_batch(self, batch_size=32):
        retarr = []

        full_chunks = []
        partial = []

        for chunk in self.chunks:
            if(len(chunk["input_ids"][0]) == 512):
                full_chunks.append(chunk)
            else:
                partial.append(chunk)
        
        if(len(full_chunks) > 0):
            batched_input = {key: torch.cat([chunk[key] for chunk in full_chunks], dim=0) for key in full_chunks[0].keys()}
        
        partial_logits = []
        with torch.no_grad():
            if(len(full_chunks) > 0):
                logits = self.model(**batched_input).logits
            if(len(partial) > 0):
                for p in partial:
                    partial_logits.append(self.model(**p).logits)
        
        if(len(full_chunks) > 0):
            retarr.extend(logits.argmax(dim=1).tolist())
        if(len(partial) > 0):
            for pl in partial_logits:
                retarr.extend(pl.argmax(dim=1).tolist())
        #logits = retarr.extend(self.model(**lastchunk)).logits
        #retarr.extend(logits.argmax(dim=1).tolist())
        if(len(retarr) > 0):
            return np.array(retarr, dtype=np.float32).mean()
        else:
            return 0

sa = SentimentAnalyzer()

with open("cleaned_text.txt","r") as f:
    txt = f.read()
    sa.add_text(txt)
    print(sa.get_sentiment_batch())

    sa.reset()
