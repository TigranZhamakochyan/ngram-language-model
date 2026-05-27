"""
ngram_lm.py
-----------
A from-scratch N-gram language model library built on pandas and NumPy.

Supports:
  - Tokenization of plain text into paragraph-aware token sequences
  - Uniform language model  (baseline)
  - Unigram language model  (frequency-based)
  - N-gram language model   (bigram, trigram, or any N ≥ 2)

Each model exposes:
  .probability(words)  → float   joint probability of a token sequence
  .sample(M)           → str     M tokens sampled from the model

Quick start
-----------
>>> from ngram_lm import tokenize, NGramLM
>>> import requests
>>> text = requests.get("https://www.gutenberg.org/files/1342/1342-0.txt").text
>>> tokens = tokenize(text)
>>> model = NGramLM(3, tokens)          # trigram model
>>> print(model.sample(30))             # generate 30 tokens
"""

import re
import time

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Text fetching
# ---------------------------------------------------------------------------

def get_book(url: str) -> str:
    """
    Download a Project Gutenberg plain-text book and return only the body
    text (everything between the START and END markers).

    Respects the Crawl-delay directive in robots.txt before fetching.

    Parameters
    ----------
    url : str
        Direct URL to the plain-text file on gutenberg.org.

    Returns
    -------
    str
        Body text with Windows line endings normalised to Unix.
    """
    robots = requests.get("https://www.gutenberg.org/robots.txt")
    delay = 0.5
    if "Crawl-delay" in robots.text:
        delay = float(robots.text.split("Crawl-delay:")[1].split("\n")[0].strip())
    time.sleep(delay)

    response = requests.get(url)
    text = response.text

    start = text.find("*** START")
    end = text.find("*** END")
    start = text.find("\n", start) + 1
    text = text[start:end]

    return text.replace("\r\n", "\n").strip()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(book_string: str) -> list:
    """
    Tokenize a book string into a flat list of tokens with paragraph markers.

    Each paragraph is wrapped by a START token (\\x02) and an END token
    (\\x03).  Words and punctuation marks are each treated as individual
    tokens, using the same split rule as Python's ``re`` module with the
    pattern ``r'\\w+|[^\\w\\s]'``.

    Parameters
    ----------
    book_string : str
        Raw text of a book (typically from ``get_book``).

    Returns
    -------
    list of str
        Token list, e.g. ['\\x02', 'It', 'was', 'the', 'best', ...  '\\x03',
        '\\x02', 'of', 'times', '.', '\\x03', ...]

    Examples
    --------
    >>> tokenize("Hello world.\\n\\nGoodbye.")
    ['\\x02', 'Hello', 'world', '.', '\\x03', '\\x02', 'Goodbye', '.', '\\x03']
    """
    book_string = book_string.strip()
    if not book_string:
        return ["\x02", "\x03"]

    paragraphs = re.split(r"\n\s*\n+", book_string)
    tokens = ["\x02"]
    for paragraph in paragraphs:
        tokens += re.findall(r"\w+|[^\w\s]", paragraph)
        tokens += ["\x03", "\x02"]

    # Drop the trailing incomplete START marker
    return tokens[:-1]


# ---------------------------------------------------------------------------
# Uniform language model
# ---------------------------------------------------------------------------

class UniformLM:
    """
    Uniform (baseline) language model.

    Assigns equal probability to every unique token seen during training,
    regardless of how often each token actually appears.

    Parameters
    ----------
    tokens : list of str
        Training token sequence produced by ``tokenize``.

    Attributes
    ----------
    mdl : pd.Series
        Maps each unique token to its uniform probability (1 / vocab_size).
    """

    def __init__(self, tokens: list):
        self.mdl = self._train(tokens)

    def _train(self, tokens: pd.Series) -> pd.Series:
        unique = pd.Series(tokens).unique()
        return pd.Series(1 / len(unique), index=unique)

    def probability(self, words: list) -> float:
        """
        Compute the joint probability of a sequence of tokens.

        Returns 0 if any token is outside the training vocabulary.

        Parameters
        ----------
        words : list of str

        Returns
        -------
        float
        """
        p = 1.0
        for word in words:
            if word not in self.mdl.index:
                return 0.0
            p *= self.mdl[word]
        return p

    def sample(self, M: int) -> str:
        """
        Sample M tokens uniformly at random and return them as a string.

        Parameters
        ----------
        M : int
            Number of tokens to sample.

        Returns
        -------
        str
            Space-joined sample.
        """
        return " ".join(np.random.choice(self.mdl.index, M, p=self.mdl.values))


# ---------------------------------------------------------------------------
# Unigram language model
# ---------------------------------------------------------------------------

class UnigramLM:
    """
    Unigram language model.

    Assigns probability proportional to each token's relative frequency in
    the training corpus.

    Parameters
    ----------
    tokens : list of str
        Training token sequence produced by ``tokenize``.

    Attributes
    ----------
    mdl : pd.Series
        Maps each unique token to its empirical probability.
    """

    def __init__(self, tokens: list):
        self.mdl = self._train(tokens)

    def _train(self, tokens: list) -> pd.Series:
        series = pd.Series(tokens)
        # Each occurrence contributes 1/N; summing per token gives frequency
        probs = pd.Series(1 / len(series), index=series)
        return probs.groupby(probs.index).sum()

    def probability(self, words: list) -> float:
        """
        Compute the joint probability of a token sequence under the unigram model.

        Returns 0 if any token is outside the training vocabulary.
        """
        p = 1.0
        for word in words:
            if word not in self.mdl.index:
                return 0.0
            p *= self.mdl[word]
        return p

    def sample(self, M: int) -> str:
        """
        Sample M tokens according to unigram frequencies.

        Parameters
        ----------
        M : int

        Returns
        -------
        str
        """
        return " ".join(np.random.choice(self.mdl.index, M, p=self.mdl.values))


# ---------------------------------------------------------------------------
# N-gram language model
# ---------------------------------------------------------------------------

class NGramLM:
    """
    N-gram language model with recursive back-off.

    Models the conditional probability P(w_n | w_{n-N+1}, ..., w_{n-1})
    using maximum-likelihood estimates from the training corpus.  For
    sequences shorter than N, the model delegates to an (N-1)-gram model
    (and so on recursively down to a UnigramLM for N=2).

    Parameters
    ----------
    N : int
        Order of the model.  Must be ≥ 2.
    tokens : list of str
        Training token sequence produced by ``tokenize``.

    Attributes
    ----------
    N : int
    ngrams : list of tuple
        All N-grams extracted from the training corpus.
    mdl : pd.DataFrame
        Columns: ``ngram``, ``n1gram`` (the (N-1)-gram prefix), ``prob``
        (conditional probability of the last token given the prefix).
    prev_mdl : UnigramLM or NGramLM
        Back-off model for sequences shorter than N.

    Examples
    --------
    >>> tokens = tokenize(get_book("https://www.gutenberg.org/files/84/84-0.txt"))
    >>> lm = NGramLM(3, tokens)          # trigram model on Frankenstein
    >>> lm.probability(["\\x02", "It", "was"])
    2.3e-07                              # illustrative value
    >>> print(lm.sample(40))
    """

    def __init__(self, N: int, tokens: list):
        if N < 2:
            raise ValueError("N must be ≥ 2")

        self.N = N
        self.ngrams = self._create_ngrams(tokens)
        self.mdl = self._train(self.ngrams)
        self.prev_mdl = UnigramLM(tokens) if N == 2 else NGramLM(N - 1, tokens)

    def _create_ngrams(self, tokens: list) -> list:
        return [
            tuple(tokens[i : i + self.N])
            for i in range(len(tokens) - self.N + 1)
        ]

    def _train(self, ngrams: list) -> pd.DataFrame:
        df = pd.DataFrame({"ngram": ngrams})
        df["n1gram"] = df["ngram"].apply(lambda x: x[:-1])

        ngram_counts = df["ngram"].value_counts()
        n1gram_counts = df["n1gram"].value_counts()

        df["prob"] = df["ngram"].apply(
            lambda x: ngram_counts[x] / n1gram_counts[x[:-1]]
        )
        return df.drop_duplicates(subset="ngram")

    def probability(self, words: list) -> float:
        """
        Compute the joint probability of a token sequence.

        Uses the chain rule:
            P(w1, ..., wk) = P(w1, ..., w_{N-1}) * prod P(w_i | w_{i-N+1}..w_{i-1})

        Delegates to prev_mdl for the initial (N-1)-gram prefix, then
        multiplies each subsequent conditional probability.

        Returns 0 for any unseen N-gram.

        Parameters
        ----------
        words : list of str

        Returns
        -------
        float
        """
        words = tuple(words)
        if len(words) < self.N:
            return self.prev_mdl.probability(words)

        p = self.prev_mdl.probability(words[: self.N - 1])
        for i in range(len(words) - self.N + 1):
            ngram = words[i : i + self.N]
            row = self.mdl[self.mdl["ngram"] == ngram]
            if row.empty:
                return 0.0
            p *= row["prob"].iloc[0]
        return p

    def sample(self, M: int) -> str:
        """
        Generate a sequence of M tokens by ancestral sampling.

        Starts from a paragraph-START marker (\\x02) and repeatedly samples
        the next token conditioned on the current context window.  If the
        context has no matches in the N-gram table, falls back to the
        (N-1)-gram model.  Appends a paragraph-END marker (\\x03) after M
        tokens and returns the full sequence as a space-joined string.

        Parameters
        ----------
        M : int
            Number of content tokens to generate (not counting the final \\x03).

        Returns
        -------
        str
            Space-joined generated sequence.
        """

        def _next_token(model, context):
            context = tuple(context)
            if len(context) < model.N - 1:
                return _next_token(model.prev_mdl, context)

            context = context[-(model.N - 1) :]
            choices = model.mdl[model.mdl["n1gram"] == context]
            if choices.empty:
                return "\x03"

            tokens = choices["ngram"].apply(lambda x: x[-1])
            probs = choices["prob"] / choices["prob"].sum()
            return np.random.choice(tokens, p=probs)

        tokens = ["\x02"]
        while len(tokens) < M:
            tokens.append(_next_token(self, tokens))
        tokens.append("\x03")
        return " ".join(tokens)
