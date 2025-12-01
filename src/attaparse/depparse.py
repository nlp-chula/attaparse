import os
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("stanza").setLevel(logging.ERROR)
logging.getLogger("stanza.models").setLevel(logging.ERROR)
logging.getLogger("stanza.models.pos").setLevel(logging.ERROR)

import stanza
from stanza.resources.common import download_models
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant
from stanza.models.common.doc import Document
from stanza.models.common import doc
from stanza.pipeline.core import Pipeline
from stanza.pipeline._constants import *
from stanza.pipeline.pos_processor import POSProcessor
from stanza.pipeline.lemma_processor import LemmaProcessor
from stanza.models.pos.data import Dataset

# Get stanza resources directory from environment or use default
STANZA_RESOURCES_DIR = os.environ.get('STANZA_RESOURCES_DIR', str(Path.home() / 'stanza_resources'))

# Track if models have been downloaded this session
_models_downloaded = False


def _ensure_models_downloaded():
    """Download models if they don't exist."""
    global _models_downloaded

    if _models_downloaded:
        return

    # Check and download base Thai models if needed
    base_model_path = os.path.join(STANZA_RESOURCES_DIR, 'th', 'tokenize', 'best.pt')
    if not os.path.exists(base_model_path):
        stanza.download('th', model_dir=STANZA_RESOURCES_DIR)

    # Check and download custom dependency parser if needed
    depparse_model_path = os.path.join(STANZA_RESOURCES_DIR, 'th', 'depparse', 'best_transformer_parser.pt')
    if not os.path.exists(depparse_model_path):
        download_list = [('depparse', 'best_transformer_parser')]
        resources = {
            'th': {
                'depparse': {
                    'best_transformer_parser': {
                        'url': "https://huggingface.co/nlp-chula/Thai-dependency-parser/resolve/main/th_best_transformer_parser_checkpoint.pt",
                        'md5': None
                    }
                }
            }
        }
        download_models(
            download_list=download_list,
            lang='th',
            resources=resources,
            model_dir=STANZA_RESOURCES_DIR,
            model_url="https://huggingface.co/nlp-chula/Thai-dependency-parser/resolve/main/th_best_transformer_parser_checkpoint.pt"
        )

    _models_downloaded = True


@register_processor_variant('pos', 'non')
class PythaiPOStag(ProcessorVariant):
    def __init__(self, config):
        pass
    def process(self, text):
        return text

@register_processor_variant('lemma', 'non')
class PythaiLemma(ProcessorVariant):
    def __init__(self, config):
        pass
    def process(self, text):
        return text

def pos_process(self, document):
    # Count tokens directly from document
    total_tokens = sum(len(sentence.tokens) for sentence in document.sentences)

    # Set dummy POS tags for all tokens
    document.set([doc.UPOS, doc.XPOS, doc.FEATS], [['.', '.', '_']] * total_tokens)
    return document

def lemma_process(self, document):
    # Count tokens directly from document
    total_tokens = sum(len(sentence.tokens) for sentence in document.sentences)

    # Set dummy lemmas for all tokens
    document.set([doc.UPOS, doc.XPOS, doc.LEMMA], [['.', '.', '.']] * total_tokens)
    return document

def load_model(tokenize_pretokenized=False):
    """
    Load the attaparse dependency parser model.

    Args:
        tokenize_pretokenized: If True, enable pretokenized mode for Stanza
                             (accepts list of token lists instead of raw text)

    Returns:
        Stanza Pipeline object
    """
    # Ensure models are downloaded before loading
    _ensure_models_downloaded()

    POSProcessor.process = pos_process
    LemmaProcessor.process = lemma_process
    nlp = stanza.Pipeline(
        lang='th',
        processors='tokenize,pos,lemma,depparse',
        dir=STANZA_RESOURCES_DIR,
        tokenize_pretokenized=tokenize_pretokenized,
        depparse_model_path=os.path.join(STANZA_RESOURCES_DIR, 'th', 'depparse', 'best_transformer_parser.pt'),
        depparse_pretrain_path=os.path.join(STANZA_RESOURCES_DIR, 'th', 'pretrain', 'fasttext157.pt'),
        depparse_forward_charlm_path=os.path.join(STANZA_RESOURCES_DIR, 'th', 'forward_charlm', 'oscar.pt'),
        depparse_backward_charlm_path=os.path.join(STANZA_RESOURCES_DIR, 'th', 'backward_charlm', 'oscar.pt'),
        use_gpu=False,
        pos_with_non=True,
        lemma_with_non=True
    )
    return nlp

def depparse(text, depparse_model):
    """
    Parse plain Thai text.

    Args:
        text: Thai text string
        depparse_model: Stanza Pipeline from load_model()

    Returns:
        Stanza Document with parsed results
    """
    token = depparse_model(text.replace(' ', ''))
    return token


def depparse_pretokenized(tokenized_sentences, depparse_model):
    """
    Parse pre-tokenized sentences.

    Args:
        tokenized_sentences: List of token lists, e.g. [["word1", "word2", ...]]
        depparse_model: Stanza Pipeline from load_model(tokenize_pretokenized=True)

    Returns:
        Stanza Document with parsed results
    """
    # Filter out blank/whitespace-only tokens from each sentence
    filtered_sentences = [
        [t for t in sentence if t.strip()]
        for sentence in tokenized_sentences
    ]
    return depparse_model(filtered_sentences)


def depparse_pipe_delimited(pipe_text, depparse_model):
    """
    Parse pipe-delimited tokenized text.

    Args:
        pipe_text: Pipe-delimited string, e.g. "word1|word2|word3"
        depparse_model: Stanza Pipeline from load_model(tokenize_pretokenized=True)

    Returns:
        Stanza Document with parsed results
    """
    # Split by pipe to get tokens
    tokens = pipe_text.split('|')

    # Filter out blank/whitespace-only tokens for consistency with internal tokeniser
    tokens = [t for t in tokens if t.strip()]

    # Wrap in list to represent single sentence
    return depparse_model([tokens])
