from unittest import TestCase


class TestCentroids(TestCase):

    def test_load_cfg(self):
        from src.cfg import DEFAULT_CFG
        assert type(DEFAULT_CFG) is dict
        assert "mode" in DEFAULT_CFG
        assert "data" in DEFAULT_CFG
        assert "model" in DEFAULT_CFG
        assert "device" in DEFAULT_CFG
        assert "episodes" in DEFAULT_CFG
        assert "patience" in DEFAULT_CFG
        assert "patience_delta" in DEFAULT_CFG
        assert "num_way" in DEFAULT_CFG
        assert "val_num_way" in DEFAULT_CFG
        assert "shot" in DEFAULT_CFG
        assert "query" in DEFAULT_CFG
        assert "iterations" in DEFAULT_CFG
        assert "adam_lr" in DEFAULT_CFG
        assert "adam_step" in DEFAULT_CFG
        assert "adam_gamma" in DEFAULT_CFG
        assert "metric" in DEFAULT_CFG
        assert "imgsz" in DEFAULT_CFG
        assert "channels" in DEFAULT_CFG
        assert "save_period" in DEFAULT_CFG
        assert "eval_each" in DEFAULT_CFG

    def test_override_cfg(self):
        from src.cfg import DEFAULT_CFG, override_cfg
        cfg = DEFAULT_CFG
        override = {"mode": "val"}
        cfg = override_cfg(cfg, override)
        assert "mode" in cfg
        assert cfg["mode"] == override["mode"]

    def test_override_cfg_nokey(self):
        from src.cfg import DEFAULT_CFG, override_cfg
        cfg = DEFAULT_CFG
        override = {"aaaaaaaaaaaaaaaaaaaa": "aaaaaaaaaaaaaaaaaaa"}
        with self.assertRaises(Exception) as context:
            cfg = override_cfg(cfg, override)