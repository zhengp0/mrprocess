"""
Model of the process, currently includes
- signal model
- linear model
"""
import os
from typing import List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import dill
import numpy as np
import matplotlib.pyplot as plt
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
    linear_cov_model_settings: dict
    linear_model_fitting_options: dict
    j_shaped: bool

    def __repr__(self):
        return f"ModelSpecs({self.name})"


@dataclass
class Bounds:
    exposure_lb: float = 0.15
    exposure_ub: float = 0.85
    draw_lb: float = 0.05
    draw_ub: float = 0.95


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
        self.linear_cov_model = LinearCovModel('signal', use_re=True,
                                               **self.specs.linear_cov_model_settings)
        self.linear_model = None

    def fit_signal_model(self):
        self.signal_model.fit_model(**self.specs.signal_model_fitting_options)
        self.df['outlier'] = (self.signal_model.w_soln <= 0.1).astype(int)[np.argsort(self.signal_data.data_id)]

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
        self.df['signal'] = signal[np.argsort(self.signal_data.data_id)]
        self.linear_data = MRData()
        self.linear_data.load_df(
            self.df[self.df.outlier == 0],
            col_obs=self.specs.col_obs,
            col_obs_se=self.specs.col_obs_se,
            col_covs=['signal'] + self.alt_cov_names + self.ref_cov_names,
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

    def get_gamma_sd(self) -> float:
        lt = self.linear_model.lt
        gamma_fisher = lt.get_gamma_fisher(lt.gamma)
        gamma_sd = 1.0/np.sqrt(gamma_fisher[0, 0])
        return gamma_sd

    def get_min_ln_rr(self) -> float:
        return np.min(self.get_prediction())

    def get_min_ln_rr_index(self) -> float:
        return np.argmin(self.get_prediction())

    def get_draws(self,
                  num_samples: int = 5000,
                  num_points: int = 100,
                  use_gamma_ub: bool = False) -> np.ndarray:
        data = self.get_prediction_data(num_points=num_points)
        beta_samples, gamma_samples = self.get_samples(num_samples=num_samples)
        if use_gamma_ub:
            gamma_samples += 2.0*self.get_gamma_sd()
        draws = self.linear_model.create_draws(data,
                                               beta_samples=beta_samples,
                                               gamma_samples=gamma_samples,
                                               random_study=True).T
        if self.specs.j_shaped:
            draws -= draws[:, self.get_min_ln_rr_index(), None]

        return draws

    def get_exposure_limits(self, bounds: Bounds = None) -> Tuple[float, float]:
        bounds = Bounds() if bounds is None else bounds
        alt_exposure = self.signal_data.get_covs(self.alt_cov_names).mean(axis=1)
        ref_exposure = self.signal_data.get_covs(self.ref_cov_names).mean(axis=1)
        exp_lower = np.quantile(ref_exposure, bounds.exposure_lb)
        exp_upper = np.quantile(alt_exposure, bounds.exposure_ub)

        return (exp_lower, exp_upper)

    def get_draw_limits(self, bounds: Bounds = None) -> Tuple[np.ndarray, np.ndarray]:
        bounds = Bounds() if bounds is None else bounds
        draws = self.get_draws()
        return (np.quantile(draws, bounds.draw_lb, axis=0),
                np.quantile(draws, bounds.draw_ub, axis=0))

    def get_effective_exp_index(self,
                                exposures: np.ndarray = None,
                                bounds: Bounds = None):
        exposures = self.get_prediction_exposures() if exposures is None else exposures
        exp_lower, exp_upper = self.get_exposure_limits(bounds)
        return (exposures >= exp_lower) & (exposures <= exp_upper)

    def is_harmful(self, draws: np.ndarray = None, bounds: Bounds = None) -> bool:
        index = self.get_effective_exp_index(bounds=bounds)
        draws = self.get_draws() if draws is None else draws
        median = np.median(draws, axis=0)
        return np.sum(median[index] >= 0) > 0.5*np.sum(index)

    def get_score(self,
                  draws: np.ndarray = None,
                  bounds: Bounds = None,
                  use_gamma_ub: bool = False) -> float:
        bounds = Bounds() if bounds is None else bounds
        index = self.get_effective_exp_index(bounds=bounds)
        draws = self.get_draws(use_gamma_ub=use_gamma_ub) if draws is None else draws
        draw_lower = np.quantile(draws, bounds.draw_lb, axis=0)
        draw_upper = np.quantile(draws, bounds.draw_ub, axis=0)
        return draw_lower[index].mean() if self.is_harmful(draws=draws) else -draw_upper[index].mean()

    def plot_data(self, ax=None):
        ax = plt.subplot() if ax is None else ax
        data = self.signal_data
        alt_exposure = data.get_covs(self.alt_cov_names)
        ref_exposure = data.get_covs(self.ref_cov_names)

        alt_mean = alt_exposure.mean(axis=1)
        ref_mean = ref_exposure.mean(axis=1)
        ref_exposures = np.repeat(self.exposure_lb, data.num_obs)
        signal = self.get_signal(
            alt_cov=[ref_mean for _ in self.alt_cov_names],
            ref_cov=[ref_exposures for _ in self.ref_cov_names]
        )
        prediction = self.linear_model.predict(MRData(covs={'signal': signal}))
        if self.specs.j_shaped:
            prediction -= self.get_min_ln_rr()

        trim_index = self.signal_model.w_soln <= 0.1
        ax.scatter(alt_mean, prediction + data.obs,
                   c='gray', s=5.0/data.obs_se, alpha=0.5)
        ax.scatter(alt_mean[trim_index], prediction[trim_index] + data.obs[trim_index],
                   c='red', marker='x', s=5.0/data.obs_se[trim_index])

    def plot_model(self,
                   bounds: Bounds = None,
                   ax=None,
                   title: str = None,
                   xlabel: str = 'exposure',
                   ylabel: str = 'ln relative risk',
                   xlim: tuple = None,
                   ylim: tuple = None,
                   xscale: str = None,
                   yscale: str = None,
                   folder: Union[str, Path] = None):
        bounds = Bounds() if bounds is None else bounds
        exp = self.get_prediction_exposures()
        exp_lower, exp_upper = self.get_exposure_limits(bounds=bounds)

        ax = plt.subplot() if ax is None else ax
        draws = self.get_draws()
        wider_draws = self.get_draws(use_gamma_ub=True)
        draws_lower = np.quantile(draws, bounds.draw_lb, axis=0)
        draws_upper = np.quantile(draws, bounds.draw_ub, axis=0)
        wider_draws_lower = np.quantile(wider_draws, bounds.draw_lb, axis=0)
        wider_draws_upper = np.quantile(wider_draws, bounds.draw_ub, axis=0)

        draws_median = np.median(draws, axis=0)

        ax.plot(exp, draws_median, color='#69b3a2', linewidth=1)
        ax.fill_between(exp, draws_lower, draws_upper, color='#69b3a2', alpha=0.2)
        ax.fill_between(exp, wider_draws_lower, wider_draws_upper, color='#69b3a2', alpha=0.2)
        ax.axvline(exp_lower, linestyle='--', color='k', linewidth=1)
        ax.axvline(exp_upper, linestyle='--', color='k', linewidth=1)
        ax.axhline(0.0, linestyle='--', color='k', linewidth=1)

        title = self.specs.name if title is None else title
        score = self.get_score(draws=draws, bounds=bounds)
        low_score = self.get_score(draws=wider_draws, bounds=bounds, use_gamma_ub=True)
        title = f"{title}: score = ({low_score: .3f}, {score: .3f})"

        ax.set_title(title, loc='left')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        self.plot_data(ax=ax)

        if folder is not None:
            folder = Path(folder)
            if not folder.exists():
                os.mkdir(folder)
            plt.savefig(folder / f"{self.specs.name}.pdf", bbox_inches='tight')

        return ax

    def save_model(self, folder: Union[str, Path]):
        folder = Path(folder)
        if not folder.exists():
            os.mkdir(folder)
        file = folder / f"{self.specs.name}.pkl"
        with open(file, "wb") as f:
            dill.dump(self, f)

    def __repr__(self):
        return f"Model({self.specs.name})"
