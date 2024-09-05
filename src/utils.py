# utility functions

import lightgbm as lgb
import pickle
from pathlib import Path
from typing import Dict, List


def get_model(model_path: Path) -> lgb.Booster:
    """Loads a LightGBM model from a pickle file and returns it.

    Parameters
    ----------
    model_path : Path
        Path to the pickle file containing the model.

    Returns
    -------
    lgb.Booster
        The loaded model.
    """
    model = pickle.load(open(model_path, "rb"))
    assert isinstance(
        model, lgb.Booster
    ), "Model is not an instance of lgb.Booster, villain."
    return model


def get_model_paths(
    run_path: str, model_type: str, suffix: str, prefix: str, model_names: List[str]
) -> Dict[str, List[Path]]:
    """Retrieves the paths of the models in a given directory based on the provided model type and
    suffix.

    Parameters:
        run_path (str): The directory containing the models.
        model_type (str): The type of model to retrieve (e.g. lgbm, dummy).
        suffix (str): The suffix of the model (e.g. full, geno_only, chem_only, dummy).
        prefix (str): The prefix of the model paths (e.g. Bloom2013_).
        model_names (List[str]): The list of models to retrieve.

    Returns:
        Dict[str, List[Path]]: A dictionary with the model names as keys and the
            paths to the models as values.
    """
    # suffix = "" if suffix == "full" else suffix

    run_path = Path(run_path)

    model_paths = {
        model: list(run_path.glob(f"{prefix}{model}*{suffix}/{model_type}.pkl"))
        for model in model_names
    }

    return model_paths
