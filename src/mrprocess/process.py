"""
Pipline that process all the pairs
"""
import os
from typing import List, Union
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from mrprocess.model import Bounds, Model, load_specs
from mrprocess.utils import read_yaml
from mrprocess.diagnostic import DataDiagnostic


class Pipeline:
    def __init__(self, ro_pairs: List[str], info: Union[str, Path, dict]):
        self.ro_pairs = ro_pairs
        self.info = read_yaml(info)
        self.model_specs = {
            ro_pair: load_specs(self.info, ro_pair)
            for ro_pair in self.ro_pairs
        }
        self.models = {
            ro_pair: Model(self.model_specs[ro_pair])
            for ro_pair in self.ro_pairs
        }

    @property
    def size(self):
        return len(self.ro_pairs)

    def fit_models(self):
        for ro_pair in self.ro_pairs:
            self.models[ro_pair].fit_signal_model()
            self.models[ro_pair].fit_linear_model()

    def plot_models(self, folder: Union[str, Path] = None):
        _, ax = plt.subplots(self.size, 1, figsize=(8, 5*self.size))
        for i, ro_pair in enumerate(self.ro_pairs):
            self.models[ro_pair].plot_model(ax=ax[i])
        if folder is not None:
            folder = Path(folder)
            if not folder.exists():
                os.mkdir(folder)
            plt.savefig(folder / "all_pairs.pdf", bbox_inches='tight')

    def save_models(self, folder: Union[str, Path]):
        for ro_pair in self.ro_pairs:
            self.models[ro_pair].save_model(folder=folder)

    def create_score(self,
                     bounds: Bounds = None,
                     folder: Union[str, Path] = None) -> pd.DataFrame:
        scores = [self.models[ro_pair].get_score(bounds=bounds)
                  for ro_pair in self.ro_pairs]
        low_scores = [self.models[ro_pair].get_score(bounds=bounds, use_gamma_ub=True)
                      for ro_pair in self.ro_pairs]
        num_studies = [self.models[ro_pair].signal_data.num_studies
                       for ro_pair in self.ro_pairs]
        gammas = [self.models[ro_pair].linear_model.gamma_soln[0]
                  for ro_pair in self.ro_pairs]
        gamma_sds = [self.models[ro_pair].get_gamma_sd()
                     for ro_pair in self.ro_pairs]
        df = pd.DataFrame({
            'ro_pair': self.ro_pairs,
            'score': scores,
            'low_score': low_scores,
            'gamma': gammas,
            'gamma_sd': gamma_sds,
            'num_studies': num_studies,
        })

        if folder is not None:
            folder = Path(folder)
            if not folder.exists():
                os.mkdir(folder)
            df.to_csv(folder / "all_pair_scores.csv", index=False)

        return df

    def run(self,
            model_folder: Union[str, Path] = None,
            plot_folder: Union[str, Path] = None,
            score_folder: Union[str, Path] = None):

        self.fit_models()
        if model_folder is not None:
            self.save_models(model_folder)
        self.plot_models(folder=plot_folder)
        df = self.create_score(folder=score_folder)
        return df
    

class DiagnosticPipeline:
    def __init__(self, ro_pairs: List[str], info: Union[str, Path, dict]):
        self.ro_pairs = ro_pairs
        self.info = read_yaml(info)
        self.model_specs = {
            ro_pair: load_specs(self.info, ro_pair)
            for ro_pair in self.ro_pairs
        }
        self.models = {
            ro_pair: DataDiagnostic(self.model_specs[ro_pair])
            for ro_pair in self.ro_pairs
        }

    @property
    def size(self):
        return len(self.ro_pairs)

    def fit_models(self):
        for ro_pair in self.ro_pairs:
            self.models[ro_pair].fit_model()

    def plot_models(self):
        _, ax = plt.subplots(self.size, 1, figsize=(8, 5*self.size))
        for i, ro_pair in enumerate(self.ro_pairs):
            self.models[ro_pair].plot_model(ax=ax[i])
    
    def plot_residuals(self):
        _, ax = plt.subplots(self.size, 1, figsize=(8, 5*self.size))
        for i, ro_pair in enumerate(self.ro_pairs):
            self.models[ro_pair].plot_residual(ax=ax[i])

    def summarize_models(self) -> pd.DataFrame:
        return pd.concat([self.models[ro_pair].summarize_model()
                          for ro_pair in self.ro_pairs])
