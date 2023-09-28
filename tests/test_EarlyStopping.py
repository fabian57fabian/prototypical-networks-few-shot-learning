import os
import shutil
from datetime import datetime

from unittest import TestCase
from src.EarlyStopping import EarlyStopping
class TestEarlyStopping(TestCase):

    def test_call_no_stop(self):
        es = EarlyStopping(5, delta=0)
        es(10)
        es(5)
        for _ in range(10):
            es(3)
        assert not es.early_stop

    def test_call_stop(self):
        es = EarlyStopping(5, delta=0)
        es(10)
        es(5)
        for _ in range(10):
            es(3)

        for _ in range(10):
            es(4)
        assert es.early_stop

    def test_call_stop_at4(self):
        patience = 4
        es = EarlyStopping(patience, delta=0)
        es(10)
        es(5)
        for _ in range(10):
            es(3)
        for _ in range(patience - 1):
            es(4)
        assert not es.early_stop
        es(4)
        assert es.early_stop

    def test_es_deactivated(self):
        es = EarlyStopping(-1, delta=0)
        es(10)
        es(5)
        for _ in range(10):
            es(3)
        for _ in range(10):
            es(4)
        assert not es.early_stop, "should not be invoked"

    def test_trace_func(self):
        called = {}
        called["state"] = False
        def trace_function(mess):
            called["state"] = True

        es = EarlyStopping(2, delta=0, trace_func=trace_function)
        es(10)
        es(1)
        es(2)
        es(2)
        es(2)
        es(2)
        assert called["state"]