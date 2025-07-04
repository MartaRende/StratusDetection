import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, List, Optional
from collections import defaultdict
from matplotlib import pyplot as plt
import os

class TransitionAnalyzer:
    """Handles critical transition detection and analysis"""
    
    def __init__(self, metrics):
        self.metrics = metrics
        
    def detect_critical_transitions(
        self,
        days: List[str],
        min_slope: float = 100,
        min_peak_distance: str = "10min",
        smooth_window: str = "15min",
        plot_day: str = "2024-11-08"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Detect critical transitions using multiple methods"""
        df = self.metrics.get_delta_btw_geneva_dole()
        days = self.metrics._normalize_days_input(days)
        df = df.sort_values("datetime").set_index("datetime")
        df = df[df.index.strftime("%Y-%m-%d").isin(days)].sort_index()

        df["expected_delta"] = df["expected_geneva"] - df["expected_dole"]
        df["predicted_delta"] = df["predicted_geneva"] - df["predicted_dole"]
        results = {}

        for prefix in ["expected", "predicted"]:
            series = df[f"{prefix}_delta"]
            smoothed = series.rolling(smooth_window, center=True).mean()
            first_deriv = smoothed.diff()
            second_deriv = first_deriv.diff()
            
            z_scores = stats.zscore(series.fillna(0))
            change_points = np.where(np.abs(z_scores) > 2.5)[0]

            min_samples = max(1, int(pd.Timedelta(min_peak_distance).total_seconds() / 
                            (df.index[1] - df.index[0]).total_seconds()))
            
            features_to_analyze = {
                'slope': first_deriv.abs().fillna(0).values,
                'accel': second_deriv.abs().fillna(0).values,
                'delta': series.abs().values
            }
            
            all_peaks = []
            for feature_name, feature_values in features_to_analyze.items():
                try:
                    peaks, _ = find_peaks(
                        feature_values,
                        height=np.percentile(feature_values, 65),
                        distance=min_samples
                    )
                    all_peaks.extend(peaks)
                except:
                    continue

            unique_peaks = list(set(list(change_points) + all_peaks))
            peaks_df = df.iloc[unique_peaks].copy() if unique_peaks else pd.DataFrame()
            
            if not peaks_df.empty:
                peaks_df["slope_magnitude"] = first_deriv.iloc[unique_peaks].abs()
                peaks_df["z_score"] = z_scores[unique_peaks]
                peaks_df["confidence"] = (
                    0.5 * peaks_df["slope_magnitude"] / peaks_df["slope_magnitude"].max() +
                    0.5 * peaks_df["z_score"].abs() / peaks_df["z_score"].abs().max()
                )
                peaks_df = peaks_df.sort_values("confidence", ascending=False)

            results[f"{prefix}_transitions"] = peaks_df

            if plot_day and (pd.to_datetime(plot_day).date() in df.index.date):
                self._plot_transition_analysis(prefix, df, plot_day, peaks_df, 
                                             first_deriv, second_deriv, z_scores)

        if not df.empty and "datetime" in df.reset_index().columns:
            df_reset = df.reset_index()
            df_reset["date"] = df_reset["datetime"].dt.date
            results["num_datetimes_per_day"] = df_reset.groupby("date")["datetime"].count().to_dict()
        else:
            results["num_datetimes_per_day"] = {}
    
        return results

    def _plot_transition_analysis(self, prefix, df, plot_day, peaks_df, 
                                first_deriv, second_deriv, z_scores):
        """Helper method to plot transition analysis"""
        day_data = df[df.index.date == pd.to_datetime(plot_day).date()]
        
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        axs[0].plot(day_data.index, day_data[f"{prefix}_geneva"], label="Geneva")
        axs[0].plot(day_data.index, day_data[f"{prefix}_dole"], label="Dole")
        axs[0].plot(day_data.index, day_data[f"{prefix}_delta"], 
                   label="Delta", color='green', linestyle='--')
        axs[0].set_title(f"{prefix.capitalize()} Values and Delta")
        axs[0].legend()
        axs[0].grid()
        
        axs[1].plot(day_data.index, first_deriv[day_data.index], 
                   label="1st Derivative (Slope)", color='orange')
        axs[1].plot(day_data.index, second_deriv[day_data.index], 
                   label="2nd Derivative (Accel)", color='red')
        axs[1].set_title("Derivatives Analysis")
        axs[1].legend()
        axs[1].grid()
        
        z_scores_series = pd.Series(z_scores, index=df.index)
        axs[2].plot(day_data.index, z_scores_series.loc[day_data.index], 
                   label="Z-scores", color='purple')
        axs[2].axhline(y=2.5, color='r', linestyle='--', label='Threshold')
        axs[2].axhline(y=-2.5, color='r', linestyle='--')
        axs[2].set_title("Statistical Change Detection")
        axs[2].legend()
        axs[2].grid()
        
        day_peaks = peaks_df[peaks_df.index.date == pd.to_datetime(plot_day).date()]
        for ax in axs:
            for idx, row in day_peaks.iterrows():
                ax.axvline(x=idx, color='r', linestyle=':', alpha=0.7)
                if ax == axs[0]:
                    ax.text(idx, ax.get_ylim()[1]*0.9, 
                           f"Conf: {row['confidence']:.2f}", 
                           rotation=90, va='top')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.metrics.save_path, f"{prefix}_transitions_{plot_day}.png"),
            dpi=self.metrics.plot_config.dpi, bbox_inches='tight'
        )
        plt.close()

    def match_strongest_peaks(
        self,
        peaks_results: Dict[str, pd.DataFrame],
        time_window: str = "3H",
        min_confidence: float = 0.8,
        include_unmatched: bool = True
    ) -> pd.DataFrame:
        """Enhanced matching of critical transitions"""
        exp_peaks = peaks_results.get("expected_transitions", pd.DataFrame()).reset_index()
        pred_peaks = peaks_results.get("predicted_transitions", pd.DataFrame()).reset_index()
        num_datetimes_per_day = peaks_results.get("num_datetimes_per_day", {})
        
        if exp_peaks.empty or pred_peaks.empty:
            return pd.DataFrame()

        if "datetime" not in exp_peaks.columns:
            exp_peaks = exp_peaks.rename(columns={exp_peaks.columns[0]: "datetime"})
        if "datetime" not in pred_peaks.columns:
            pred_peaks = pred_peaks.rename(columns={pred_peaks.columns[0]: "datetime"})

        max_conf = max(exp_peaks["confidence"].max(), pred_peaks["confidence"].max())
        exp_peaks["norm_conf"] = exp_peaks["confidence"] / max_conf
        pred_peaks["norm_conf"] = pred_peaks["confidence"] / max_conf

        matches = []
        max_time_diff = pd.Timedelta(time_window)
        used_pred_indices = set()
        matched_exp_indices = set()

        for _, exp_row in exp_peaks.iterrows():
            candidates = pred_peaks[
                (pred_peaks["datetime"].between(
                    exp_row["datetime"] - max_time_diff,
                    exp_row["datetime"] + max_time_diff
                )) &
                (~pred_peaks.index.isin(used_pred_indices))
            ].copy()

            if not candidates.empty:
                candidates["time_diff"] = (candidates["datetime"] - exp_row["datetime"]).abs().dt.total_seconds()
                candidates["time_score"] = 1 - (candidates["time_diff"] / max_time_diff.total_seconds())
                candidates["conf_score"] = 1 - abs(candidates["norm_conf"] - exp_row["norm_conf"])
                
                if "expected_delta" in exp_row and "predicted_delta" in candidates.columns:
                    exp_delta = exp_row["expected_delta"] if not pd.isnull(exp_row["expected_delta"]) else 0
                    candidates["delta_diff"] = (candidates["predicted_delta"] - exp_delta).abs()
                    max_delta_diff = candidates["delta_diff"].max() if candidates["delta_diff"].max() > 0 else 1
                    candidates["delta_score"] = 1 - (candidates["delta_diff"] / max_delta_diff)
                else:
                    candidates["delta_score"] = 1.0

                candidates["combined_score"] = (
                    0.2 * candidates["time_score"] +
                    0.2 * candidates["conf_score"] +
                    0.6 * candidates["delta_score"]
                )
            
                best_match = candidates.nlargest(1, "combined_score").iloc[0]

                if best_match["combined_score"] >= min_confidence:
                    matches.append({
                        "expected_time": exp_row["datetime"],
                        "predicted_time": best_match["datetime"],
                        "expected_confidence": exp_row["confidence"],
                        "predicted_confidence": best_match["confidence"],
                        "time_difference_sec": (best_match["datetime"] - exp_row["datetime"]).total_seconds(),
                        "confidence_similarity": best_match["conf_score"],
                        "combined_score": best_match["combined_score"],
                        "match_status": "matched",
                        "is_early": best_match["datetime"] < exp_row["datetime"],
                        "is_late": best_match["datetime"] > exp_row["datetime"],
                        "expected_slope": exp_row.get("slope_magnitude", None),
                        "predicted_slope": best_match.get("slope_magnitude", None),
                        "expected_z_score": exp_row.get("z_score", None),
                        "predicted_z_score": best_match.get("z_score", None)
                    })
                    matched_exp_indices.add(exp_row.name)
                    used_pred_indices.add(best_match.name)

        if include_unmatched:
            unmatched_exp = exp_peaks[
                (~exp_peaks.index.isin(matched_exp_indices)) &
                (exp_peaks["confidence"] >= min_confidence)
            ]
            for _, exp_row in unmatched_exp.iterrows():
                matches.append({
                    "expected_time": exp_row["datetime"],
                    "predicted_time": pd.NaT,
                    "expected_confidence": exp_row["confidence"],
                    "predicted_confidence": None,
                    "time_difference_sec": None,
                    "confidence_similarity": None,
                    "combined_score": None,
                    "match_status": "unmatched_expected",
                    "is_early": None,
                    "is_late": None,
                    "expected_slope": exp_row.get("slope_magnitude", None),
                    "predicted_slope": None,
                    "expected_z_score": exp_row.get("z_score", None),
                    "predicted_z_score": None
                })

        unmatched_pred = pred_peaks[
            (~pred_peaks.index.isin(used_pred_indices)) &
            (pred_peaks["confidence"] >= min_confidence)
        ]
        for _, pred_row in unmatched_pred.iterrows():
            matches.append({
                "expected_time": pd.NaT,
                "predicted_time": pred_row["datetime"],
                "expected_confidence": None,
                "predicted_confidence": pred_row["confidence"],
                "time_difference_sec": None,
                "confidence_similarity": None,
                "combined_score": None,
                "match_status": "unmatched_predicted",
                "is_early": None,
                "is_late": None,
                "expected_slope": None,
                "predicted_slope": pred_row.get("slope_magnitude", None),
                "expected_z_score": None,
                "predicted_z_score": pred_row.get("z_score", None)
            })

        result_df = pd.DataFrame(matches)
        result_df = result_df.sort_values(
            by=["combined_score", "expected_time"], 
            ascending=[False, True],
            na_position="last"
        )

        result_df = result_df[
            (result_df["time_difference_sec"].isna()) | (result_df["time_difference_sec"] >= 0)
        ]
     
        if not result_df.empty and "expected_time" in result_df.columns:
            result_df["expected_date"] = pd.to_datetime(result_df["expected_time"]).dt.date
            result_df["num_datetimes_for_day"] = result_df["expected_date"].map(num_datetimes_per_day).fillna(0).astype(int)
            result_df = result_df.drop(columns=["expected_date"])
        
        return result_df