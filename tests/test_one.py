import pytest
import torch
from mlops_classifier.models.model import MyAwesomeModel



class TestClass:
    def test_mytest(self):
        assert 1 == 1
        assert 2 == 2
        assert MyAwesomeModel() is not None

    import os.path
    @pytest.mark.skipif(not os.path.exists("data/processed/corruptmnist/test.pt"), reason="Data files not found")
    def test_mytest2(self):
        model = MyAwesomeModel()
        test_set = torch.load("data/processed/corruptmnist/test.pt")
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)
        with torch.no_grad():
            for images, labels in test_loader:
                assert model(images) is not None
                break

    def test_kek(self):
        assert 1 == 1
        assert 2 == 2
