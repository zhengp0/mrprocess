"""
Utility functions
"""
from typing import Union
from pathlib import Path
import yaml
import pandas as pd
from scipy.stats import norm


def read_csv(file: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(file, (str, Path)):
        file = pd.read_csv(file)
    else:
        assert isinstance(file, pd.DataFrame), "Wrong input type for read_csv."
    return file


def read_yaml(file: Union[str, Path, dict]) -> dict:
    if isinstance(file, (str, Path)):
        with open(file) as f:
            file = yaml.load(f, Loader=yaml.FullLoader)
    else:
        assert isinstance(file, dict), "Wrong input type for read_yaml."
    return file


def get_p_val(mean: float, sd: float) -> float:
    p = norm.cdf(0.0, loc=mean, scale=sd)
    return min(p, 1 - p)
