"""
Model of the process, currently includes
- signal model
- linear model
"""
from typing import List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import dill
import numpy as np
from .utils import read_csv, read_yaml
from mrtool import MRData, LinearCovModel, LogCovModel, MRBRT
from mrtool.core.other_sampling import sample_simple_lme_beta


@dataclass
class ModelSpecs:
    name: str
    data_path: Union[str, Path]
    col_obs: str
    col_obs_se: str
    col_covs: List[str]
    col_study_id: str
    signal_cov_model_type: str
    signal_cov_model_settings: dict
    signal_model_inlier_pct: float
    signal_model_fitting_options: dict
    linear_model_fitting_options: dict

    def __repr__(self):
        return f"ModelSpecs({self.name})"


def load_specs(info: Union[str, Path, dict], name: str) -> ModelSpecs:
    all_info = read_yaml(info)
    members = {**all_info['default'], **all_info[name]}
    return ModelSpecs(name=name, **members)


class Model:
    def __init__(self, specs: ModelSpecs):
        self.specs = specs

        self.df = read_csv(specs.data_path)
        self.df['data_id'] = np.arange(self.df.shape[0])

        self.signal_data = MRData()
        self.signal_data.load_df(
            self.df,
            col_obs=self.specs.col_obs,
            col_obs_se=self.specs.col_obs_se,
            col_covs=self.specs.col_covs,
            col_study_id=self.specs.col_study_id,
            col_data_id='data_id'
        )
        if self.specs.signal_cov_model_type == 'linear':
            CovModel = LinearCovModel
        elif self.specs.signal_cov_model_type == 'log':
            CovModel = LogCovModel
        else:
            raise ValueError("unknown signal cov model type.")

        self.signal_cov_model = CovModel(**self.specs.signal_cov_model_settings)
        self.signal_model = MRBRT(self.signal_data,
                                  cov_models=[self.signal_cov_model],
                                  inlier_pct=self.specs.signal_model_inlier_pct)

        # exposure infomation
        self.alt_cov_names = self.signal_cov_model.alt_cov
        self.ref_cov_names = self.signal_cov_model.ref_cov
        self.exposures = self.signal_data.get_covs(self.alt_cov_names +
                                                   self.ref_cov_names)
        self.exposure_lb = self.exposures.min()
        self.exposure_ub = self.exposures.max()

        # place holders
        self.linear_data = None
        self.linear_cov_model = LinearCovModel('signal', use_re=True)
        self.linear_model = None

    def fit_signal_model(self):
        self.signal_model.fit_model(**self.specs.signal_model_fitting_options)
        self.df['outlier'] = (self.signal_model.w_soln <= 0.1).astype(int)[self.signal_data.data_id]

    def get_signal(self,
                   alt_cov: List[np.ndarray],
                   ref_cov: List[np.ndarray]) -> np.ndarray:
        covs = {}
        for i, cov_name in enumerate(self.alt_cov_names):
            covs[cov_name] = alt_cov[i]
        for i, cov_name in enumerate(self.ref_cov_names):
            covs[cov_name] = ref_cov[i]
        data = MRData(covs=covs)
        return self.signal_model.predict(data)

    def fit_linear_model(self):
        signal = self.get_signal(
            alt_cov=[self.signal_data.covs[cov_name] for cov_name in self.alt_cov_names],
            ref_cov=[self.signal_data.covs[cov_name] for cov_name in self.ref_cov_names]
        )
        self.df['signal'] = signal[self.signal_data.data_id]
        self.linear_data = MRData()
        self.linear_data.load_df(
            self.df[self.df.outlier == 0],
            col_obs=self.specs.col_obs,
            col_obs_se=self.specs.col_obs_se,
            col_covs=['signal'],
            col_study_id=self.specs.col_study_id,
            col_data_id='data_id'
        )
        self.linear_model = MRBRT(self.linear_data, cov_models=[self.linear_cov_model])
        self.linear_model.fit_model(**self.specs.linear_model_fitting_options)

    def get_beta_samples(self, num_samples: int) -> np.ndarray:
        return sample_simple_lme_beta(num_samples, self.linear_model)

    def get_gamma_samples(self, num_samples: int) -> np.ndarray:
        return np.repeat(self.linear_model.gamma_soln.reshape(1, 1),
                         num_samples, axis=0)

    def get_samples(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_beta_samples(num_samples), self.get_gamma_samples(num_samples)

    def get_prediction_exposures(self, num_points: int = 100):
        return np.linspace(self.exposure_lb, self.exposure_ub, num_points)

    def get_prediction_data(self, num_points: int = 100) -> MRData:
        exposures = self.get_prediction_exposures(num_points=num_points)
        ref_exposures = np.repeat(self.exposure_lb, num_points)
        signal = self.get_signal(
            alt_cov=[exposures for _ in self.alt_cov_names],
            ref_cov=[ref_exposures for _ in self.ref_cov_names]
        )
        return MRData(covs={'signal': signal})

    def get_prediction(self, num_points: int = 100) -> np.ndarray:
        data = self.get_prediction_data(num_points=num_points)
        return self.linear_model.predict(data)

    def get_draws(self,
                  num_samples: int = 1000,
                  num_points: int = 100) -> np.ndarray:
        data = self.get_prediction_data(num_points=num_points)
        beta_samples, gamma_samples = self.get_samples(num_samples=num_samples)
        return self.linear_model.create_draws(data,
                                              beta_samples=beta_samples,
                                              gamma_samples=gamma_samples,
                                              random_study=True).T

    def save_model(self, folder: Union[str, Path]):
        file = Path(folder) / f"{self.specs.name}.pkl"
        with open(file, "wb") as f:
            dill.dump(self, f)

    def __repr__(self):
        return f"Model({self.specs.name})"
