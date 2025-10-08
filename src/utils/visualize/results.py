import json
import pickle as pkl
from pathlib import Path
from typing import Callable, List, Self
import seaborn as sns
from matplotlib import colors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from scipy.integrate import solve_ivp

from utils.ode import map_equation


class ODEVisualizer:
    def __init__(
        self: Self,
        fun: Callable,
        x0: np.ndarray,
        t: np.ndarray,
        labels: List[str] | None = None,
    ) -> None:
        self.fun = fun
        self.x0 = x0
        self.t = t
        self.solution = None
        self.labels = (
            labels if labels is not None else [f"Var {i + 1}" for i in range(len(x0))]
        )

    def solve(
        self: Self, method: str = "LSODA", rtol: float = 1e-12, atol: float = 1e-12
    ) -> Self:
        self.solution = solve_ivp(
            self.fun,
            (self.t[0], self.t[-1]),
            self.x0,
            method=method,
            t_eval=self.t,
            rtol=rtol,
            atol=atol,
        )

        return self

    def plot_solution(self: Self) -> None:
        if self.solution is None:
            raise ValueError("No solution found. Please run the solve() method first.")

        num_vars = self.solution.y.shape[0]
        fig, axs = plt.subplots(num_vars, 1, figsize=(10, 5 * num_vars))

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i in range(num_vars):
            axs[i].plot(
                self.solution.t, self.solution.y[i], label=f"{self.labels[i]} Solution"
            )
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel(self.labels[i])
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


class ODEResultVisualizer(ODEVisualizer):
    def __init__(
        self: Self,
        fun: Callable,
        x0: np.ndarray,
        t: np.ndarray,
        result_dir: str,
        labels: List[str] | None = None,
    ) -> None:
        super().__init__(fun, x0, t, labels)

        self.result_dir = Path(result_dir)
        if not self.result_dir.exists():
            Exception(f"Result directory {self.result_dir} does not exist.")

        combined_result_file = self.result_dir / "combined_results.json"
        if not combined_result_file.exists():
            Exception(f"Combined result file {combined_result_file} does not exist.")

        with open(combined_result_file, "r") as f:
            data = json.load(f)

        results = []
        for model, values in data.items():
            result = {"model": model}
            result.update(values["metrics"])
            result.update(values["parameters"])
            results.append(result)

        self.results_df = pd.DataFrame(results)
        self.results_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def get_best_model(
        self: Self,
        metric: str,
        minimize: bool = True,
        fixed_params: dict | None = None,
    ) -> str:
        if metric not in self.results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        df = self.results_df.copy()
        if fixed_params:
            for param, value in fixed_params.items():
                if param in df.columns:
                    df = df[df[param] == value]

                    if df.empty:
                        raise ValueError(
                            f"No models found with {param} = {value} in results."
                        )
                    elif not isinstance(df, pd.DataFrame):
                        raise ValueError(
                            f"Filtering by {param} = {value} did not return a DataFrame."
                        )
                else:
                    raise ValueError(f"Parameter {param} not found in results.")
        if df.empty:
            raise ValueError("No models available to evaluate.")

        if minimize:
            best_model_row = df.loc[df[metric].idxmin()]
        else:
            best_model_row = df.loc[df[metric].idxmax()]

        return best_model_row

    def plot_metric_distribution(
        self: Self,
        metric: str,
        save: bool = False,
    ) -> None:
        if metric not in self.results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        plt.figure(figsize=(10, 6))
        plt.hist(self.results_df[metric], bins=30, alpha=0.7, color="blue")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {metric}")
        plt.grid()

        if save:
            fig_path = f"distribution_{metric}.png"
            plt.savefig(fig_path)
            print(f"Distribution plot saved to {fig_path}")

        plt.show()

    def plot_model_solution(
        self: Self,
        model_name: str,
        save: bool = False,
    ) -> None:
        if self.solution is None:
            raise ValueError("No solution found. Please run the solve() method first.")

        model_file = self.result_dir / f"{model_name}.pkl"
        if not model_file.exists():
            raise ValueError(f"Model file {model_file} does not exist.")

        with open(model_file, "rb") as f:
            model = pkl.load(f)

        solution_sim = model.simulate(self.x0, self.t).T

        num_vars = solution_sim.shape[0]
        # Temporary code to save each subplot separately
        for i in range(num_vars):
            plt.figure(figsize=(10, 5))
            plt.plot(
                self.solution.t,
                self.solution.y[i],
                label=f"{self.labels[i]} True",
                color="blue",
            )
            plt.plot(
                self.t,
                solution_sim[i],
                label=f"{self.labels[i]} Simulated",
                linestyle="--",
                color="red",
            )
            plt.xlabel("Time")
            plt.ylabel(self.labels[i])
            plt.legend()
            plt.grid()
            plt.title(f"Solution Comparison for {self.labels[i]} using {model_name}")

            if save:
                fig_path = f"solution_{model_name}_{self.labels[i]}.png"
                plt.savefig(fig_path)
                print(f"Solution plot saved to {fig_path}")

            plt.show()
        # fig, axs = plt.subplots(num_vars, 1, figsize=(10, 5 * num_vars))
        # if not isinstance(axs, np.ndarray):
        #     axs = [axs]
        # for i in range(num_vars):
        #     axs[i].plot(
        #         self.solution.t,
        #         self.solution.y[i],
        #         label=f"{self.labels[i]} True",
        #         color="blue",
        #     )
        #     axs[i].plot(
        #         self.t,
        #         solution_sim[i],
        #         label=f"{self.labels[i]} Simulated",
        #         linestyle="--",
        #         color="red",
        #     )
        #     axs[i].set_xlabel("Time")
        #     axs[i].set_ylabel(self.labels[i])
        #     axs[i].legend()
        #     axs[i].grid()
        # plt.tight_layout()
        #
        # if save:
        #     fig_path = f"solution_{model_name}.png"
        #     plt.savefig(fig_path)
        #     print(f"Solution plot saved to {fig_path}")
        #
        # plt.show()

    def plot_best_model_solution(
        self: Self,
        metric: str,
        minimize: bool = True,
        fixed_params: dict | None = None,
        save: bool = False,
    ) -> None:
        if metric not in self.results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        df = self.results_df.copy()
        if fixed_params:
            for param, value in fixed_params.items():
                if param in df.columns:
                    df = df[df[param] == value]

                    if df.empty:
                        raise ValueError(
                            f"No models found with {param} = {value} in results."
                        )
                    elif not isinstance(df, pd.DataFrame):
                        raise ValueError(
                            f"Filtering by {param} = {value} did not return a DataFrame."
                        )
                else:
                    raise ValueError(f"Parameter {param} not found in results.")
        if df.empty:
            raise ValueError("No models available to evaluate.")

        if minimize:
            best_model_row = df.loc[df[metric].idxmin()]
        else:
            best_model_row = df.loc[df[metric].idxmax()]

        best_model_name = best_model_row["model"]
        self.plot_model_solution(best_model_name, save=save)

    def plot_coefficients(
        self: Self,
        model_name: str,
        save: bool = False,
    ) -> None:
        model_file = self.result_dir / f"{model_name}.pkl"
        if not model_file.exists():
            raise ValueError(f"Model file {model_file} does not exist.")

        with open(model_file, "rb") as f:
            model = pkl.load(f)

        if not hasattr(model, "coefficients"):
            raise ValueError(f"Model {model_name} does not have coefficients to plot.")

        true_coef = map_equation(self.fun, model.feature_library)
        learned_coef = model.coefficients()

        # --- 3. Set up Shared Color Scale ---
        # Find the maximum absolute value across BOTH arrays to center the color map
        abs_max = max(np.abs(true_coef).max(), np.abs(learned_coef).max())

        # Set vmin and vmax to be symmetric around zero.
        vmin = -abs_max
        vmax = abs_max

        # Create the symmetric log normalizer
        # linthresh: The range within which the plot is linear (-linthresh to +linthresh)
        # A smaller value makes the log scale apply to smaller numbers.
        linthresh = 1e-2
        log_norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        sns.heatmap(
            true_coef,
            cmap="coolwarm",
            cbar=True,
            ax=axs[0],
            yticklabels=[f"{label}" for label in self.labels],
            xticklabels=model.feature_library.get_feature_names_out(),
            norm=log_norm,
        )
        axs[0].set_title("True Coefficients")
        sns.heatmap(
            learned_coef,
            cmap="coolwarm",
            cbar=True,
            ax=axs[1],
            yticklabels=[f"{label}" for label in self.labels],
            xticklabels=model.feature_library.get_feature_names_out(),
            norm=log_norm,
        )
        axs[1].set_title(f"Learned Coefficients of {model_name}")
        plt.tight_layout()

        if save:
            fig_path = f"coefficients_{model_name}.png"
            plt.savefig(fig_path)
            print(f"Coefficient plot saved to {fig_path}")

        plt.show()

    def plot_best_model_coefficients(
        self: Self,
        metric: str,
        minimize: bool = True,
        fixed_params: dict | None = None,
        save: bool = False,
    ) -> None:
        if metric not in self.results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        df = self.results_df.copy()
        if fixed_params:
            for param, value in fixed_params.items():
                if param in df.columns:
                    df = df[df[param] == value]

                    if df.empty:
                        raise ValueError(
                            f"No models found with {param} = {value} in results."
                        )
                    elif not isinstance(df, pd.DataFrame):
                        raise ValueError(
                            f"Filtering by {param} = {value} did not return a DataFrame."
                        )
                else:
                    raise ValueError(f"Parameter {param} not found in results.")
        if df.empty:
            raise ValueError("No models available to evaluate.")

        if minimize:
            best_model_row = df.loc[df[metric].idxmin()]
        else:
            best_model_row = df.loc[df[metric].idxmax()]

        best_model_name = best_model_row["model"]
        self.plot_coefficients(best_model_name, save=save)

    def plot_parameter_heatmap(
        self: Self,
        param_x: str,
        param_y: str,
        metric: str,
        agg_func: str = "mean",
        fixed_params: dict | None = None,
        save: bool = False,
    ) -> None:
        """
        Plot a heatmap of a performance metric over two parameters.
        Parameters are not expected to be evenly spaced as they are sampled
        """
        if param_x not in self.results_df.columns:
            raise ValueError(f"Parameter {param_x} not found in results.")
        if param_y not in self.results_df.columns:
            raise ValueError(f"Parameter {param_y} not found in results.")
        if metric not in self.results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        df = self.results_df.copy()
        if fixed_params:
            for param, value in fixed_params.items():
                if param in df.columns:
                    df = df[df[param] == value]

                    if df.empty:
                        raise ValueError(
                            f"No models found with {param} = {value} in results."
                        )
                    elif not isinstance(df, pd.DataFrame):
                        raise ValueError(
                            f"Filtering by {param} = {value} did not return a DataFrame."
                        )
                else:
                    raise ValueError(f"Parameter {param} not found in results.")
        if df.empty:
            raise ValueError("No models available to evaluate.")

        # Aggregate the metric over the two parameters
        if agg_func in ["mean", "median", "min", "max"]:
            # pivot_table = df.pivot_table(
            #     index=param_y, columns=param_x, values=metric, aggfunc=agg_func
            # )
            pivot_table = df.groupby([param_y, param_x])[metric].agg(agg_func).unstack()
        else:
            raise ValueError(f"Aggregation function {agg_func} is not supported.")

        # Plot the heatmap with model names annotated
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": metric},
        )
        plt.title(f"Heatmap of {metric} ({agg_func}) over {param_x} and {param_y}")
        plt.xlabel(param_x)
        plt.ylabel(param_y)

        if save:
            fig_path = f"heatmap_{param_x}_{param_y}_{metric}.png"
            plt.savefig(fig_path)
            print(f"Heatmap saved to {fig_path}")

        plt.show()
