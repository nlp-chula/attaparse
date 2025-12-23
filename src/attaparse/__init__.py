from .depparse import (
    load_model,
    depparse,
    depparse_pretokenized,
    depparse_pipe_delimited,
    STANZA_RESOURCES_DIR
)

__version__ = "1.0.1"

__all__ = [
    'load_model',
    'depparse',
    'depparse_pretokenized',
    'depparse_pipe_delimited',
    'STANZA_RESOURCES_DIR'
]