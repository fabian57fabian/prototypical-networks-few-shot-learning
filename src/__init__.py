from src.cfg import DEFAULT_CFG, override_cfg

__version__ = "1.1.0"


def entrypoint(overrides):
    if "mode" not in overrides:
        raise Exception("please set a mode from train | eval | learn_centroids | predict")
    cfg = override_cfg(DEFAULT_CFG, overrides)
    if cfg["mode"] == "train":
        from src.core import meta_train
        return meta_train(cfg)
    elif cfg["mode"] == "eval":
        from src.core import meta_test
        return meta_test(cfg)
    elif cfg["mode"] == "learn":
        from src.core import learn
        return learn(cfg)
    elif cfg["mode"] == "predict":
        from src.core import predict
        return predict(cfg)
    else:
        raise Exception(f"Unknown mode {cfg['mode']}")
