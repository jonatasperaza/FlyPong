#!/usr/bin/env python3
"""Le um checkpoint do PyTorch (formato zip) sem o PyTorch, devolvendo arrays numpy.

Usado para os parametros publicados do flyvis (Lappalainen et al. 2024,
licenca MIT). So entende tensores densos e colecoes simples (dict, list,
OrderedDict), que e o que esses checkpoints contem.
"""
from __future__ import annotations

import collections
import pickle
import sys
import zipfile

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor_v2(storage, offset, size, stride, *args):
    if not size:
        return storage[offset].copy()
    itemsize = storage.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)).copy()


def _rebuild_parameter(data, requires_grad, backward_hooks, *args):
    return data


def load(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    pkl = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl[: -len("data.pkl")]

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_parameter":
                return _rebuild_parameter
            if module == "torch" and name.endswith("Storage"):
                return _StorageType(name)
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module.startswith("torch"):
                raise pickle.UnpicklingError(f"tipo nao suportado: {module}.{name}")
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, stype, key, _location, _numel = pid
            raw = z.read(f"{prefix}data/{key}")
            return np.frombuffer(raw, dtype=stype.dtype)

    return U(z.open(pkl)).load()


if __name__ == "__main__":
    obj = load(sys.argv[1])

    def show(o, ind=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, np.ndarray):
                    print(f"{ind}{k}: {v.dtype} {v.shape}")
                else:
                    print(f"{ind}{k}: {type(v).__name__}")
                    show(v, ind + "  ")
    show(obj)
