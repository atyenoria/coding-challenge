import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter
import matplotlib.pyplot as plt

class SimpleTokenizer:
    def __init__(self):
        self.vocab = set([
            # Common words
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "is", "are", "was", "were", "will", "would", "could", "should",
            "I", "you", "he", "she", "it", "we", "they",
            # Nouns
            "cat", "dog", "house", "tree", "car", "book", "city", "world",
            "bird", "fish", "boy", "girl", "man", "woman", "child",
            # Verbs
            "run", "jump", "eat", "sleep", "read", "write", "speak", "think",
            "walk", "see", "hear", "feel", "like", "love", "hate",
            # Adjectives
            "big", "small", "happy", "sad", "fast", "slow", "good", "bad",
            "hot", "cold", "new", "old", "young", "tall", "short",
            # Punctuation
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
        
        # Define common word patterns for more realistic text
        self.patterns = {
            "the": ["cat", "dog", "house", "tree", "car", "book", "bird", "fish"],
            "is": ["big", "small", "happy", "sad", "good", "bad", "hot", "cold"],
            "was": ["running", "jumping", "reading", "sleeping", "walking"],
            "cat": ["is", "was", "and", "likes", "runs", "sleeps"],
            "dog": ["is", "was", "and", "likes", "runs", "barks"],
            "they": ["are", "were", "will", "could", "should", "might"],
            "i": ["am", "was", "will", "could", "should", "might"],
            "a": ["big", "small", "happy", "sad", "good", "bad", "new", "old"],
        }
        
        # Add more sophisticated transitions
        self.context_patterns = {
            ("the", "cat"): ["is", "was", "likes", "runs"],
            ("is", "very"): ["happy", "sad", "big", "small", "good", "bad"],
            ("they", "are"): ["happy", "sad", "good", "bad", "running", "sleeping"],
        }
    
    def get_next_token_probs(self, input_ids: List[int]) -> np.ndarray:
        probs = np.ones(self.vocab_size) * 0.01  # Base probability
        
        if input_ids:
            last_word = self.tokenizer.id2token[input_ids[-1]]
            
            # Single token patterns
            if last_word in self.patterns:
                for word in self.patterns[last_word]:
                    if word in self.tokenizer.token2id:
                        probs[self.tokenizer.token2id[word]] = 0.3
            
            # Context-based patterns (last two tokens)
            if len(input_ids) >= 2:
                last_two = tuple(self.tokenizer.id2token[id] for id in input_ids[-2:])
                if last_two in self.context_patterns:
                    for word in self.context_patterns[last_two]:
                        if word in self.tokenizer.token2id:
                            probs[self.tokenizer.token2id[word]] = 0.4
            
            # Add controlled randomness
            probs += np.random.uniform(0, 0.1, size=self.vocab_size)
            
            # Reduce immediate repetition
            if len(input_ids) > 1:
                probs[input_ids[-1]] *= 0.1
        
        return probs / probs.sum()

class EnhancedGreedyDecoder:
    def __init__(self, model: MockLanguageModel, tokenizer: SimpleTokenizer, 
                 max_length: int = 20, temperature: float = 1.0,
                 repetition_penalty: float = 1.2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        # Track generation statistics
        self.stats = {
            "total_tokens": 0,
            "unique_tokens": 0,
            "repetitions": 0,
            "pattern_matches": 0
        }
    
    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """ Apply temperature scaling to adjust randomness"""
        logits = np.log(probs + 1e-10)  # Add epsilon for numerical stability
        logits = logits / self.temperature
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for stability
        return exp_logits / exp_logits.sum()
    
    def _apply_repetition_penalty(self, probs: np.ndarray, input_ids: List[int]) -> np.ndarray:
        """Penalize recently used tokens"""
        penalty_window = min(10, len(input_ids))  # Look at last 10 tokens
        recent_tokens = set(input_ids[-penalty_window:])
        
        penalized_probs = probs.copy()
        for token in recent_tokens:
            penalized_probs[token] = penalized_probs[token] / self.repetition_penalty
        
        return penalized_probs / penalized_probs.sum()
    
    def _get_next_token(self, input_ids: List[int]) -> int:
        """Get next token with temperature and repetition penalty"""
        # Get base probabilities
        probs = self.model.get_next_token_probs(input_ids)
        
        # Apply repetition penalty
        probs = self._apply_repetition_penalty(probs, input_ids)
        
        # Apply temperature
        probs = self._apply_temperature(probs)
        
        # Sample from distribution
        return int(np.random.choice(len(probs), p=probs))
    
    def generate(self, prompt: str, min_length: int = 5) -> str:
        """Generate text with enhanced features"""
        input_ids = self.tokenizer.encode(prompt)
        
        # Reset statistics
        self.stats = {
            "total_tokens": len(input_ids),
            "unique_tokens": len(set(input_ids)),
            "repetitions": 0,
            "pattern_matches": 0
        }
        
        consecutive_repeats = 0
        max_consecutive_repeats = 3
        
        while len(input_ids) < self.max_length:
            next_token = self._get_next_token(input_ids)
            
            # Update statistics
            self.stats["total_tokens"] += 1
            self.stats["unique_tokens"] = len(set(input_ids + [next_token]))
            
            # Check for repetition
            if len(input_ids) > 0 and next_token == input_ids[-1]:
                consecutive_repeats += 1
                self.stats["repetitions"] += 1
                if consecutive_repeats >= max_consecutive_repeats:
                    break
            else:
                consecutive_repeats = 0
            
            # Check for pattern matches
            if len(input_ids) > 0:
                last_word = self.tokenizer.id2token[input_ids[-1]]
                next_word = self.tokenizer.id2token[next_token]
                if last_word in self.model.patterns and next_word in self.model.patterns[last_word]:
                    self.stats["pattern_matches"] += 1
            
            input_ids.append(next_token)
            
            # Stop conditions
            if len(input_ids) >= min_length:
                # Stop on punctuation after minimum length
                if next_token in [self.tokenizer.token2id["."], 
                                self.tokenizer.token2id["!"],
                                self.tokenizer.token2id["?"]]:
                    break
            
            # Force stop on EOS
            if next_token == self.tokenizer.token2id[self.tokenizer.special_tokens["eos"]]:
                break
        
        return self.tokenizer.decode(input_ids)

def visualize_generations(decoder: EnhancedGreedyDecoder, prompt: str, num_samples: int = 5):
    """Visualize multiple generations with statistics"""
    generations = []
    stats = []
    
    plt.figure(figsize=(15, 10))
    
    # Generate samples
    for i in range(num_samples):
        text = decoder.generate(prompt)
        generations.append(text)
        stats.append(decoder.stats.copy())
    
    # Plot statistics
    metrics = ["total_tokens", "unique_tokens", "repetitions", "pattern_matches"]
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [s[metric] for s in stats]
        plt.bar(range(num_samples), values)
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.xlabel("Sample")
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()
    
    # Print generations
    print("\nGenerated Samples:")
    print("-" * 50)
    for i, text in enumerate(generations):
        print(f"Sample {i+1}: {text}")

def main():
    # Initialize components
    tokenizer = SimpleTokenizer()
    model = MockLanguageModel(tokenizer)
    
    # Test different configurations
    print("Testing Different Configurations:")
    print("-" * 50)
    
    configs = [
        ("Conservative", {"temperature": 0.5, "repetition_penalty": 1.5}),
        ("Balanced", {"temperature": 1.0, "repetition_penalty": 1.2}),
        ("Creative", {"temperature": 1.5, "repetition_penalty": 1.1}),
    ]
    
    prompts = [
        "the cat is",
        "i feel",
        "they were",
        "a small dog"
    ]
    
    for name, params in configs:
        print(f"\n{name} Configuration:")
        decoder = EnhancedGreedyDecoder(model, tokenizer, **params)
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            for _ in range(3):  # Generate 3 samples
                output = decoder.generate(prompt)
                print(f"Generated: {output}")
    
    # Visualize generations for one configuration
    print("\nDetailed Analysis of Balanced Configuration:")
    balanced_decoder = EnhancedGreedyDecoder(model, tokenizer, 
                                           temperature=1.0, 
                                           repetition_penalty=1.2)
    
    visualize_generations(balanced_decoder, "the cat is", num_samples=5)

if __name__ == "__main__":
    main()