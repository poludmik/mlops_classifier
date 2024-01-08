import pytest

import os.path
@pytest.mark.skipif(not os.path.exists("data/"), reason="Data files not found")
def test_something_about_data():
    assert 1 == 1
    assert os.path.exists("data/") == True
