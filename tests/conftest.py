import os, sys, shutil, tempfile, pytest
from lore_memory.engine import Engine, Config

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="sq3_")
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def engine(tmp_dir):
    e = Engine(Config(data_dir=tmp_dir, embedding_dims=64))
    yield e
    e.close()


@pytest.fixture
def grammar_engine(tmp_dir):
    """Engine pinned to the legacy hand-rolled grammar parser.

    Use this when a test asserts grammar-specific output (exact predicate
    names like 'detest', 'relocate_to'). The default `engine` fixture
    uses spaCy, which lemmatizes and structures differently.
    """
    e = Engine(Config(data_dir=tmp_dir, embedding_dims=64, use_spacy=False))
    yield e
    e.close()
