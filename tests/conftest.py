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
