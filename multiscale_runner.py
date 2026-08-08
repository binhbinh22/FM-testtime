import numpy as np
import torch

from run import ERA5Config, ERA5DataLoader, ChronosProposer
from era5_physics_energy import ERA5PhysicsEnergyScorer
from multiscale_search import MultiscaleSearchConfig, MultiscaleTimeSearchEngine


class MultiscaleERA5Evaluator:
    """Thin wrapper that exposes the new multiscale pipeline on top of the existing code."""

    def __init__(self, config: ERA5Config):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)
        self.scorer = ERA5PhysicsEnergyScorer(dt=config.interval_hours * 3600)
        self.engine = MultiscaleTimeSearchEngine(
            proposer=self.forecaster,
            scorer=self.scorer,
            config=MultiscaleSearchConfig(
                n_scenarios_coarse=8,
                n_scenarios_intermediate=8,
                n_scenarios_fine=8,
                k_coarse=3,
                k_intermediate=3,
                k_fine=1,
                dtw_threshold=1.5,
                lambda_activity=1.0,
            ),
        )

    def run_window(self, start_idx: int):
        data = self.loader.get_test_window(start_idx)
        history = data["history_t2m"]
        gt_t2m = data["gt_t2m"]
        u10 = data["u10"]
        v10 = data["v10"]

        result = self.engine.run_multiscale_search(
            history=history,
            u_wind_future=u10,
            v_wind_future=v10,
            prediction_steps=self.config.prediction_steps,
            ground_truth=gt_t2m,
        )

        print("\n=== Multiscale Search Result ===")
        print("Final candidate shape:", result["final_candidate"].shape)
        print("Final index:", result["final_index"])
        if result["metrics"]:
            print("RMSE (norm):", round(result["metrics"]["rmse_norm"], 6))
            print("RMSE (denorm):", round(result["metrics"]["rmse_denorm"], 6))

        plot_paths = self.engine.save_multiscale_visuals(
            history=history,
            ground_truth=gt_t2m,
            result=result,
            lat_idx=self.config.inspect_lat,
            lon_idx=self.config.inspect_lon,
            output_dir=".",
        )
        print("Saved plots:")
        for key, path in plot_paths.items():
            print(f"- {key}: {path}")

        return result


if __name__ == "__main__":
    config = ERA5Config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    evaluator = MultiscaleERA5Evaluator(config)
    evaluator.run_window(start_idx=0)
