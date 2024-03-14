from typing import List

import numpy as np
import sys
import os

def save(names: List[str], data: List[np.array], labels: List[List[str]], expname: str = None) -> None:
    expname = expname or sys.argv[0][:-3]
    os.makedirs(expname, exist_ok=True)

    assert len(names) == len(data)

    for n, d in zip(names, data):
        np.save(os.path.join(expname, n), d)