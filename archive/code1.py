"""
LLM Greedy Decoder Enhancement Challenge
Time: 3 hours

Task: Complete the EnhancedGreedyDecoder class to improve text generation quality.

Core Concepts:
1. Temperature (creativity control):
   - 0.1: Very focused, deterministic
   - 1.0: Balanced randomness
   - 2.0: Very creative/random

2. Repetition Penalty:
   - 1.0: No penalty
   - 1.2: Mild penalty
   - 2.0: Strong penalty

Example Probability Adjustments:
Original: [0.7, 0.2, 0.1] for ["cat", "dog", "bird"]
With Temperature=0.5: [0.84, 0.12, 0.04] (more focused)
With Temperature=2.0: [0.55, 0.27, 0.18] (more random)
"""

import numpy as np
from typing import List, Dict
import re
from collections import Counter
import matplotlib.pyplot as plt

# Basic components (already implemented)
class SimpleTokenizer:
    def __init__(self):
        self.vocab = set([
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "is", "are", "was", "were", "will", "would", "could", "should",
            "I", "you", "he", "she", "it", "we", "they",
            "cat", "dog", "house", "tree", "car", "book", "city", "world",
            "run", "jump", "eat", "sleep", "read", "write", "speak", "think",
            "big", "small", "happy", "sad", "fast", "slow", "good", "bad",
            ",", ".", "!", "?", " "
        ])
        self.special_tokens = {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<bos>",
            "eos": "<eos>"
        }
        self.vocab.update(self.special_tokens.values())
        self._create_mappings()
    
    def _create_mappings(self):
        self.token2id = {token: idx for idx, token in enumerate(sorted(self.vocab))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
    def encode(self, text: str) -> List[int]:
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.token2id.get(token, self.token2id[self.special_tokens["unk"]]) 
                for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token.get(id, self.special_tokens["unk"]) for id in ids]
        text = " ".join(tokens)
        return re.sub(r'\s+([,.!?])', r'\1', text)

class MockLanguageModel:
    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.token2id)
        self.patterns = {
            "the": ["cat", "dog", "house", "tree", "car", "book"],
            "is": ["big", "small", "happy", "sad", "good", "bad"],
            "was": ["running", "jumping", "reading", "sleeping"],
            "cat": ["is", "was", "and", "likes"],
            "dog": ["is", "was", "and", "likes"],
            "they": ["are", "were", "will", "could"],
        }
    
    def get_next_token_probs(self, input_ids: List[int]) -> np.ndarray:
        probs = np.ones(self.vocab_size) * 0.01
        if input_ids:
            last_word = self.tokenizer.id2token[input_ids[-1]]
            if last_word in self.patterns:
                for word in self.patterns[last_word]:
                    if word in self.tokenizer.token2id:
                        probs[self.tokenizer.token2id[word]] = 0.3
            probs += np.random.uniform(0, 0.1, size=self.vocab_size)
        return probs / probs.sum()

class GreedyDecoder:
    def __init__(self, model: MockLanguageModel, tokenizer: SimpleTokenizer, max_length: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _get_next_token(self, input_ids: List[int]) -> int:
        probs = self.model.get_next_token_probs(input_ids)
        return int(np.argmax(probs))
    
    def generate(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt)
        while len(input_ids) < self.max_length:
            next_token = self._get_next_token(input_ids)
            input_ids.append(next_token)
            if next_token == self.tokenizer.token2id[self.tokenizer.special_tokens["eos"]]:
                break
        return self.tokenizer.decode(input_ids)

# Your task: Complete this class
class EnhancedGreedyDecoder(GreedyDecoder):
    def __init__(self, model: MockLanguageModel, tokenizer: SimpleTokenizer, 
                 max_length: int = 20, temperature: float = 1.0,
                 repetition_penalty: float = 1.2):
        super().__init__(model, tokenizer, max_length)
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
    
    def _apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """TODO: Implement temperature scaling
        1. Convert probabilities to logits
        2. Apply temperature
        3. Convert back to probabilities
        """
        pass
    
    def _apply_repetition_penalty(self, probs: np.ndarray, input_ids: List[int]) -> np.ndarray:
        """TODO: Implement repetition penalty
        1. Identify repeated tokens
        2. Apply penalty
        3. Renormalize
        """
        pass
    
    def _get_next_token(self, input_ids: List[int]) -> int:
        """TODO: Implement enhanced token selection"""
        pass
    
    def generate(self, prompt: str) -> str:
        """TODO: Implement enhanced generation"""
        pass

def test_decoder():
    tokenizer = SimpleTokenizer()
    model = MockLanguageModel(tokenizer)
    
    # Test basic decoder
    basic_decoder = GreedyDecoder(model, tokenizer)
    print("Basic Decoder Output:")
    print(basic_decoder.generate("the cat"))
    
    # Test your enhanced decoder
    enhanced_decoder = EnhancedGreedyDecoder(model, tokenizer)
    print("\nEnhanced Decoder Output (should be better):")
    print(enhanced_decoder.generate("the cat"))

if __name__ == "__main__":
    test_decoder()