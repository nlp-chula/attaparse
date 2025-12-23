# Attaparse : Thai Dependency Parser

`attaparse` is a Thai dependency parser trained using [stanza](https://github.com/stanfordnlp/stanza/tree/main). Attaparse uses [PhayaThaiBERT](https://huggingface.co/clicknext/phayathaibert) as a based model in training process. The model refer to **Stanza*P with no POS** model in [Thai Universal Dependency Treebank (TUD)](https://github.com/nlp-chula/TUD).

## Content
1. [Installation](#installation)
2. [Usage](#usage)

## Installation

`attaparse` can be installed using `pip`:
```
pip install attaparse
```

## Usage

### Initialising

```python
from attaparse import load_model, depparse

nlp = load_model()
```

#### Device Selection

By default, attaparse runs on CPU. To use GPU acceleration:

```python
# NVIDIA GPU
nlp = load_model(device="cuda")

# Apple Silicon (M1/M2/M3)
nlp = load_model(device="mps")
```

### Plain Text

Uses Stanza's default Thai tokeniser.

```python
text = 'ฉันอยากกินข้าวที่แม่ทำ'

doc = depparse(text, nlp)
```

### Pipe-Delimited Input

```python
from attaparse import depparse_pipe_delimited

nlp = load_model(tokenize_pretokenized=True)
pipe_text = "ฉัน|รัก|เธอ"

doc = depparse_pipe_delimited(pipe_text, nlp)
```

### Pre-tokenised List Input

```python
from attaparse import depparse_pretokenized

nlp = load_model(tokenize_pretokenized=True)
tokens = [["ฉัน", "กิน", "ข้าว"]]

doc = depparse_pretokenized(tokens, nlp)
```

### Access the Results

```python
print(f'\n{text}\n',*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
```
- `.id` : the id of the word.
- `.head` : the head of the word.
- `.deprel` : the dependency relationship between the word and the head.

## Citation

If you use `attaparse` in your project or publication, please cite as follows:

Panyut Sriwirote, Wei Qi Leong, Charin Polpanumas, Santhawat Thanyawong, William Chandra Tjhi, Wirote Aroonmanakun, and Attapol T. Rutherford. 2025. The Thai Universal Dependency Treebank. Transactions of the Association for Computational Linguistics, 13:376–391.

*BibTex*

```
@article{sriwirote-etal-2025-thai,
    title = "The {T}hai {U}niversal {D}ependency Treebank",
    author = "Sriwirote, Panyut  and
      Leong, Wei Qi  and
      Polpanumas, Charin  and
      Thanyawong, Santhawat  and
      Tjhi, William Chandra  and
      Aroonmanakun, Wirote  and
      Rutherford, Attapol T.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "13",
    year = "2025",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2025.tacl-1.18/",
    doi = "10.1162/tacl_a_00745",
    pages = "376--391"
}
```
