"""
Diagnostic module
"""
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mrtool import MRBRT, LinearCovModel, LogCovModel, MRData
from mrtool.core.other_sampling import (extract_simple_lme_hessian,
                                        extract_simple_lme_specs)
from pandas import DataFrame
from scipy.stats import norm

from mrprocess.model import ModelSpecs
from mrprocess.utils import get_p_val, read_csv


class DataDiagnostic:
    """Data diagnostic class
    """

    def __init__(self, specs: ModelSpecs):
        self.specs = specs
        self.df = None
        self.ref_names = None
        self.alt_names = None

        self.exp_bounds = None
        self.exp_limits = None

        self.lrr_pred = None
        self.exp_pred = None

        self.signal_model = None
        self.linear_model = None
        self.linear_model_fill = None

        self.se_model = {}
        self.num_fill = 0

        self.score = None
        self.score_fill = None
        self.outer_draws = None
        self.inner_draws = None
        self.outer_draws_fill = None
        self.inner_draws_fill = None

    def fit_model(self):
        # process data
        self.process_data()

        # fit signal model
        self.signal_model = self.fit_signal_model(self.df)

        # fit linear model
        self.linear_model = self.fit_linear_model(self.df[self.df.outlier == 0])

        # get score
        self.score, self.inner_draws, self.outer_draws = self.get_score(self.linear_model)

        self.detect_pub_bias()

        if self.has_pub_bias:
            self.adjust_pub_bias()

    def process_data(self):
        # load data frame
        self.df = read_csv(self.specs.data_path)
        # create data_id
        self.df["data_id"] = np.arange(self.df.shape[0])
        # convert study_id to string
        self.df[self.specs.col_study_id] = self.df[self.specs.col_study_id].astype(str)
        # create exposure names
        ref_cov = self.specs.signal_cov_model_settings["ref_cov"]
        alt_cov = self.specs.signal_cov_model_settings["alt_cov"]
        self.ref_names = ref_cov if isinstance(ref_cov, list) else [ref_cov]
        self.alt_names = alt_cov if isinstance(alt_cov, list) else [alt_cov]
        # get exposure information
        ref_exp = self.df[self.ref_names].to_numpy()
        alt_exp = self.df[self.alt_names].to_numpy()
        # create mid points for reference and alternative exposures
        self.df["ref_mid"] = ref_exp.mean(axis=1)
        self.df["alt_mid"] = alt_exp.mean(axis=1)
        # get the upper and lower bounds
        exp_min = min(ref_exp.min(), alt_exp.min())
        exp_max = max(ref_exp.max(), alt_exp.max())
        self.exp_bounds = (exp_min, exp_max)
        self.exp_pred = np.linspace(exp_min, exp_max, 100)
        # get the exposure limits for the score estimation
        exp_lb = np.quantile(self.df.ref_mid, 0.15)
        exp_ub = np.quantile(self.df.alt_mid, 0.85)
        self.exp_limits = (exp_lb, exp_ub)

    def fit_signal_model(self, df: DataFrame) -> MRBRT:
        # create data for signal model
        signal_data = MRData()
        signal_data.load_df(
            df,
            col_obs=self.specs.col_obs,
            col_obs_se=self.specs.col_obs_se,
            col_covs=self.specs.col_covs,
            col_study_id=self.specs.col_study_id,
            col_data_id='data_id'
        )

        # create covariate model for signal model
        if self.specs.signal_cov_model_type == "linear":
            signal_cov_model = LinearCovModel
        elif self.specs.signal_cov_model_type == "log":
            signal_cov_model = LogCovModel
        else:
            raise ValueError("unknown cov model type.")
        signal_cov_models = [signal_cov_model(**self.specs.signal_cov_model_settings)]

        # create signal model
        signal_model = MRBRT(signal_data,
                             cov_models=signal_cov_models,
                             inlier_pct=self.specs.signal_model_inlier_pct)

        # fit signal model
        signal_model.fit_model(**self.specs.signal_model_fitting_options)

        # mark outliers
        outlier_index = (signal_model.w_soln <= 0.1).astype(int)
        self.df["outlier"] = outlier_index[np.argsort(signal_model.data.data_id)]

        # create signal
        signal = signal_model.predict(signal_model.data)
        self.df['signal'] = signal[np.argsort(signal_model.data.data_id)]

        # create prediction at the reference mid points
        ref_cov = np.repeat(self.exp_bounds[0], self.df.shape[0])
        alt_cov = self.df.ref_mid.to_numpy()
        data = MRData(covs={**{name: ref_cov for name in self.ref_names},
                            **{name: alt_cov for name in self.alt_names}})
        pred = signal_model.predict(data)
        self.df["ref_pred"] = pred
        self.df["alt_pred"] = pred + self.df[self.specs.col_obs]

        # create prediction at the prediction exposures
        ref_cov = np.repeat(self.exp_bounds[0], self.exp_pred.size)
        alt_cov = self.exp_pred
        data = MRData(covs={**{name: ref_cov for name in self.ref_names},
                            **{name: alt_cov for name in self.alt_names}})
        self.lrr_pred = signal_model.predict(data)

        return signal_model

    def fit_linear_model(self, df: DataFrame) -> MRBRT:
        linear_data = MRData()
        linear_data.load_df(
            df,
            col_obs=self.specs.col_obs,
            col_obs_se=self.specs.col_obs_se,
            col_covs=["signal"],
            col_study_id=self.specs.col_study_id,
            col_data_id="data_id"
        )

        # create covariate model for linear model
        linear_cov_models = [
            LinearCovModel("signal",
                           use_re=True,
                           **self.specs.linear_cov_model_settings)
        ]

        # create linear model
        linear_model = MRBRT(linear_data, cov_models=linear_cov_models)

        # fit linear model
        linear_model.fit_model(**self.specs.linear_model_fitting_options)

        return linear_model

    def detect_pub_bias(self):
        # compute inflated obs_se
        self.df["inflated_obs_se"] = np.sqrt(
            self.df[self.specs.col_obs_se]**2 +
            self.df["signal"]**2*self.linear_model.gamma_soln[0]
        )

        # compute the residual
        self.df["residual"] = self.df[self.specs.col_obs] - self.df["signal"]
        self.df["weighted_residual"] = self.df["residual"]/self.df["inflated_obs_se"]

        # compute the correlation between residual and obs_se
        r = self.df.weighted_residual[self.df.outlier == 0]
        self.se_model["mean"] = np.mean(r)
        self.se_model["sd"] = max(1, np.std(r))/np.sqrt(r.size)
        self.se_model["pval"] = get_p_val(self.se_model["mean"],
                                          self.se_model["sd"])

    @property
    def has_pub_bias(self) -> bool:
        return self.se_model["pval"] < 0.05

    def adjust_pub_bias(self):
        # get residual
        df = self.df[self.df.outlier == 0].reset_index(drop=True)
        residual = df.residual.to_numpy()
        # compute rank of the absolute residual
        rank = np.zeros(residual.size, dtype=int)
        rank[np.argsort(np.abs(residual))] = np.arange(1, residual.size + 1)
        # get the furthest residual according to the sign of egger regression
        sort_index = np.argsort(residual)
        if self.se_model["mean"] > 0.0:
            sort_index = sort_index[::-1]
        # compute the number of data points need to be filled
        self.num_fill = residual.size - rank[sort_index[-1]]
        # get data to fill
        df_fill = df.iloc[sort_index[:self.num_fill], :]
        df_fill[self.specs.col_study_id] = "fill_" + df_fill[self.specs.col_study_id]
        df_fill["data_id"] = self.df.shape[0] + np.arange(self.num_fill)
        df_fill["residual"] = -df_fill["residual"]
        df_fill[self.specs.col_obs] = df_fill["signal"] + df_fill["residual"]
        df_fill["alt_pred"] = df_fill[self.specs.col_obs] + df_fill["ref_pred"]
        # combine with filled dataframe
        self.df = pd.concat([self.df, df_fill])
        # refit linear model
        self.linear_model_fill = self.fit_linear_model(self.df[self.df.outlier == 0])
        # compute score
        self.score_fill, self.inner_draws_fill, self.outer_draws_fill = self.get_score(self.linear_model_fill)

    def get_score(self, linear_model: MRBRT) -> Tuple[float, np.ndarray, np.ndarray]:
        # compute the posterior standard error of the fixed effect
        inner_fe_sd = np.sqrt(get_beta_sd(linear_model)**2 + linear_model.gamma_soln[0])
        outer_fe_sd = np.sqrt(get_beta_sd(linear_model)**2 +
                              linear_model.gamma_soln[0] + 2.0*get_gamma_sd(linear_model))

        # compute the lower and upper signal multiplier
        inner_betas = (norm.ppf(0.05, loc=1.0, scale=inner_fe_sd),
                       norm.ppf(0.95, loc=1.0, scale=inner_fe_sd))
        outer_betas = (norm.ppf(0.05, loc=1.0, scale=outer_fe_sd),
                       norm.ppf(0.95, loc=1.0, scale=outer_fe_sd))

        # compute the lower and upper draws
        inner_draws = np.vstack([inner_betas[0]*self.lrr_pred,
                                 inner_betas[1]*self.lrr_pred])
        outer_draws = np.vstack([outer_betas[0]*self.lrr_pred,
                                 outer_betas[1]*self.lrr_pred])

        # compute and score
        index = (self.exp_pred >= self.exp_limits[0]) & (self.exp_pred <= self.exp_limits[1])
        sign = np.sign(self.lrr_pred[index].mean())
        score = np.min(outer_draws[:, index].mean(axis=1)*sign)

        return score, inner_draws, outer_draws

    def summarize_model(self) -> DataFrame:
        summary = {
            "ro_pair": self.specs.name,
            "num_fill": self.num_fill,
            "gamma": self.linear_model.gamma_soln[0],
            "gamma_sd": get_gamma_sd(self.linear_model),
            "score": self.score
        }
        if self.has_pub_bias:
            summary.update({
                "gamma_adjusted": self.linear_model_fill.gamma_soln[0],
                "gamma_sd_adjusted": get_gamma_sd(self.linear_model_fill),
                "score_adjusted": self.score_fill
            })
        summary.update({"egger_" + name: value
                        for name, value in self.se_model.items()})
        return DataFrame(summary, index=[0])

    def plot_residual(self, ax: plt.Axes) -> plt.Axes:
        # compute the residual and observation standard deviation
        residual = self.df["residual"]
        obs_se = self.df["inflated_obs_se"]
        max_obs_se = np.quantile(obs_se, 0.99)
        fill_index = self.df[self.specs.col_study_id].str.contains("fill")

        # create funnel plot
        ax = plt.subplots()[1] if ax is None else ax
        ax.set_ylim(max_obs_se, 0.0)
        ax.scatter(residual, obs_se, color="gray", alpha=0.4)
        if fill_index.sum() > 0:
            ax.scatter(residual[fill_index], obs_se[fill_index], color="#008080", alpha=0.7)
        ax.scatter(residual[self.df.outlier == 1], obs_se[self.df.outlier == 1],
                   color='red', marker='x', alpha=0.4)
        ax.fill_betweenx([0.0, max_obs_se],
                         [0.0, -1.96*max_obs_se],
                         [0.0, 1.96*max_obs_se],
                         color='#B0E0E6', alpha=0.4)
        ax.plot([0, -1.96*max_obs_se], [0.0, max_obs_se], linewidth=1, color='#87CEFA')
        ax.plot([0.0, 1.96*max_obs_se], [0.0, max_obs_se], linewidth=1, color='#87CEFA')
        ax.axvline(0.0, color='k', linewidth=1, linestyle='--')
        ax.set_xlabel("residual")
        ax.set_ylabel("ln_rr_se")
        ax.set_title(
            f"{self.specs.name}: egger_mean={self.se_model['mean']: .3f}, "
            f"egger_sd={self.se_model['sd']: .3f}, "
            f"egger_pval={self.se_model['pval']: .3f}",
            loc="left")
        return ax

    def plot_model(self, ax: plt.Axes) -> plt.Axes:
        # plot data
        ax.scatter(self.df.alt_mid,
                   self.df.alt_pred,
                   s=5.0/self.df[self.specs.col_obs_se],
                   color="gray", alpha=0.5)
        index = self.df.outlier == 1
        ax.scatter(self.df.alt_mid[index],
                   self.df.alt_pred[index],
                   s=5.0/self.df[self.specs.col_obs_se][index],
                   marker="x", color="red", alpha=0.5)

        # plot prediction
        ax.plot(self.exp_pred, self.lrr_pred, color="#008080", linewidth=1)

        # plot uncertainties
        ax.fill_between(self.exp_pred,
                        self.inner_draws[0],
                        self.inner_draws[1], color="#69b3a2", alpha=0.2)
        ax.fill_between(self.exp_pred,
                        self.outer_draws[0],
                        self.outer_draws[1], color="#69b3a2", alpha=0.2)

        # plot filled model
        if self.num_fill > 0:
            ax.plot(self.exp_pred, self.outer_draws_fill[0],
                    linestyle="--", color="gray", alpha=0.5)
            ax.plot(self.exp_pred, self.outer_draws_fill[1],
                    linestyle="--", color="gray", alpha=0.5)
            index = self.df[self.specs.col_study_id].str.contains("fill")
            ax.scatter(self.df.alt_mid[index],
                       self.df.alt_pred[index],
                       s=5.0/self.df[self.specs.col_obs_se][index],
                       marker="o", color="#008080", alpha=0.5)

        # plot bounds
        for b in self.exp_limits:
            ax.axvline(b, linestyle="--", linewidth=1, color="k")

        # plot 0 line
        ax.axhline(0.0, linestyle="-", linewidth=1, color="k")

        # title
        title = f"{self.specs.name}: score={self.score: .3f}"
        if self.num_fill > 0:
            title = title + f", score_fill={self.score_fill: .3f}"
        ax.set_title(title, loc="left")

        return ax


def get_gamma_sd(model: MRBRT) -> float:
    gamma = model.gamma_soln
    gamma_fisher = model.lt.get_gamma_fisher(gamma)
    return 1.0/np.sqrt(gamma_fisher[0, 0])


def get_beta_sd(model: MRBRT) -> float:
    model_specs = extract_simple_lme_specs(model)
    beta_hessian = extract_simple_lme_hessian(model_specs)
    return 1.0/np.sqrt(beta_hessian[0, 0])
