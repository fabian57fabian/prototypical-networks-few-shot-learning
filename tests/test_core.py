import os
import uuid
import shutil
from PIL import Image
from unittest import TestCase
from src.core import get_allowed_base_datasets_names


class TestCore(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_allowed_base_datasets_names(self):
        ad = get_allowed_base_datasets_names()
        assert type(ad) is list
        assert len(ad) == 3
