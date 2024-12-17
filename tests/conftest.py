import os
import pytest

@pytest.fixture(autouse=True)
def setup_vcr():
    """Ensure VCR cassettes directory exists"""
    os.makedirs("tests/fixtures/vcr_cassettes", exist_ok=True)
