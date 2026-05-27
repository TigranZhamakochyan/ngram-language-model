# N-gram Language Model

A lightweight, from-scratch N-gram language model library built with Python, pandas, and NumPy. Given any plain-text corpus, the library can estimate token probabilities and generate new text by sampling from learned distributions.

## What it does

The library implements three model tiers:

| Model | What it learns |
|---|---|
| `UniformLM` | Assigns equal probability to every token in the vocabulary (baseline) |
| `UnigramLM` | Learns per-token frequencies from the corpus |
| `NGramLM` | Learns conditional probabilities P(word \| N-1 preceding words); falls back recursively to lower-order models for unseen contexts |

`NGramLM` supports any order N ≥ 2 — bigram (N=2), trigram (N=3), and beyond. Text generation uses ancestral sampling: starting from a paragraph-start marker, it repeatedly samples the next token conditioned on the current context window.

## Quick start

```python
from ngram_lm import get_book, tokenize, NGramLM

# Download a book from Project Gutenberg (respects robots.txt crawl delay)
text = get_book("https://www.gutenberg.org/files/84/84-0.txt")  # Frankenstein

# Tokenize into paragraph-aware token sequences
tokens = tokenize(text)

# Train a trigram model
model = NGramLM(N=3, tokens=tokens)

# Generate 50 tokens of new text
print(model.sample(50))

# Compute the probability of a token sequence
p = model.probability(["\x02", "It", "was", "a", "dark"])
print(f"Sequence probability: {p:.2e}")
```

Example output (trigram model trained on *Frankenstein*):
```
\x02 I had determined , if you had not come to me , I should have gone to you \x03 \x02 We rested a few hours in a hut , when I perceived the wind rise \x03
```

## Tokenization

Text is split into paragraph-aware sequences using paragraph-break delimiters (`\n\n`). Each paragraph is wrapped with:
- `\x02` — start-of-paragraph token
- `\x03` — end-of-paragraph token

Within each paragraph, words and punctuation marks are each treated as separate tokens.

## Installation

No special installation needed. Clone the repo and install the dependencies:

```bash
git clone https://github.com/TigranZhamakochyan/ngram-language-model
cd ngram-language-model
pip install pandas numpy requests
```

## Usage examples

### Comparing models on the same text

```python
from ngram_lm import tokenize, UniformLM, UnigramLM, NGramLM

tokens = tokenize(open("my_corpus.txt").read())

uniform = UniformLM(tokens)
unigram = UnigramLM(tokens)
trigram = NGramLM(3, tokens)

sequence = ["\x02", "The", "creature"]

print(uniform.probability(sequence))   # same for all sequences of this length
print(unigram.probability(sequence))   # based on individual token frequencies
print(trigram.probability(sequence))   # based on conditional context
```

### Generating text at different orders

```python
bigram  = NGramLM(2, tokens)
trigram = NGramLM(3, tokens)
fourgram = NGramLM(4, tokens)

print("Bigram: ",   bigram.sample(30))
print("Trigram: ",  trigram.sample(30))
print("4-gram: ",   fourgram.sample(30))
```

Higher N → more locally coherent output, but more sparse probability estimates and a higher chance of reproducing exact training phrases.

## Design notes

- **Recursive back-off**: `NGramLM(N)` automatically builds and stores an `NGramLM(N-1)` (down to `UnigramLM` at N=2). Probability and sampling both fall back to the lower-order model when the current context is unseen.
- **pandas-backed**: The conditional probability table is a `pd.DataFrame` with columns `ngram`, `n1gram`, and `prob`, making it easy to inspect and filter.
- **No smoothing**: This implementation uses raw maximum-likelihood estimates. For production use, consider adding Laplace or Kneser-Ney smoothing for better handling of unseen N-grams.

## Project Gutenberg

`get_book(url)` strips the standard Gutenberg header and footer, leaving only the body text. It reads the `Crawl-delay` directive from `robots.txt` and sleeps accordingly before fetching.

Some books to try:

| Book | URL |
|---|---|
| *Frankenstein* | `https://www.gutenberg.org/files/84/84-0.txt` |
| *Pride and Prejudice* | `https://www.gutenberg.org/files/1342/1342-0.txt` |
| *Moby Dick* | `https://www.gutenberg.org/files/2701/2701-0.txt` |
| *The Picture of Dorian Gray* | `https://www.gutenberg.org/files/174/174-0.txt` |

## Tech stack

- Python 3.9+
- pandas
- NumPy
- requests
