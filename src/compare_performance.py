# Compare performance of any two model

import pickle

# from os import listdir
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

import hydra
import numpy as np
from omegaconf import DictConfig
import polars as pl
import polars.selectors as cs
from rich import print

# import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from typing import Dict, List

load_dotenv()


def get_preds(
    path_of_model: Union[str, Path], data_path: Union[str, Path], run_type: str
) -> pl.DataFrame:
    """A function that generates predictions using a trained model.

    Parameters:
        model_path (str): Path to the trained model file.
        data_path (str): Path to the data file for prediction.

    Returns:
        pl.DataFrame: A DataFrame containing the actual Phenotype, predicted Phenotype, and Condition.
    """
    # print("We got here!")
    model = pickle.load(open(path_of_model, "rb"))
    data = pl.read_ipc(data_path)

    match run_type:
        case "full":
            X = data.drop(["Condition", "Strain", "Phenotype"]).to_pandas()
        case "geno_only":
            X = data.select(cs.starts_with("Y")).to_pandas()
        case "chem_only":
            X = data.select(cs.starts_with("latent")).to_pandas()
        case "dummy":
            X = data.drop(["Condition", "Strain", "Phenotype"]).to_pandas()
    y = data["Phenotype"].to_numpy()

    preds = model.predict(X)
    preds = np.where(preds > 0.5, 1, 0)

    return pl.DataFrame(
        {"Phenotype": y, "Preds": preds, "Condition": data["Condition"]}
    )


def get_preds_kfold(
    paths_of_model: Union[str, Path.glob], data_path, run_type
) -> pl.DataFrame:
    """A function that generates predictions using a trained model.

    Parameters:
        paths_of_model (str): Path to the trained model file.
        data_path (str): Path to the data file for prediction.

    Returns:
        pl.DataFrame: A DataFrame containing the actual Phenotype, predicted Phenotype, and Condition.
    """
    preds = []
    for i, path_of_model in enumerate(paths_of_model):
        result = get_preds(path_of_model, data_path, run_type)
        result = result.with_columns(pl.lit(i).alias("Fold"))
        preds.append(result)
    return pl.concat(preds)


def eval_model(pred_df: pl.DataFrame) -> pl.DataFrame:
    """Evaluates the performance of a model based on its predictions.

    Parameters:
        pred_df (pl.DataFrame): A DataFrame containing the predicted values,
            actual values, and conditions.

    Returns:
        pl.DataFrame: A DataFrame containing the accuracy, f1 score, auc roc score,
            and mathews correlation coefficient for each condition and fold.
    """
    result_df = pred_df.group_by("Condition", "Fold", maintain_order=True).map_groups(
        lambda x: pl.DataFrame(
            {
                "Condition": x["Condition"].unique(maintain_order=True),
                "Fold": x["Fold"].unique(maintain_order=True),
                "accuracy": accuracy_score(x["Phenotype"], x["Preds"]),
                "f1": f1_score(x["Phenotype"], x["Preds"]),
                "auc_roc": roc_auc_score(x["Phenotype"], x["Preds"]),
                "mathews": matthews_corrcoef(x["Phenotype"], x["Preds"]),
            }
        )
    )

    return result_df


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


def get_results(conf: DictConfig):
    """Runs the models in the `run_path` directory on the data in the `data_paths` dictionary, and
    saves the results to the `out_path` directory.

    Parameters:
        conf (DictConfig): A configuration object with the following required
            keys:
                - run_path (Path): The path to the directory containing the models.
                - data_paths (Dict[str, Path]): A dictionary mapping data names to
                    paths to the data files.
                - out_path (Path): The path to the directory to save the results to.
                - model_types (List[str]): A list of model types to run.
                - run_type (str): The suffix of the model name to distinguish between
                    different runs (e.g. 'full', 'geno_only', 'chem_only', 'dummy').

    Returns:
        None
    """
    # run_path = Path(conf.run_path)
    # data_path = Path(conf.data_paths)
    out_path = Path(conf.out_path)
    model_type = conf.model_load_keys.model_type
    model_paths = get_model_paths(**conf.model_load_keys)

    assert (
        len(conf.model_load_keys.model_names) == len(conf.data_paths)
    ), f"Mismatch between data {len(conf.data_paths)} and model {len(conf.model_load_keys.model_names)} and model names"

    print(model_paths)

    for model_name, model_path in model_paths.items():
        for data_name, data_path in conf.data_paths.items():
            print(model_name, data_name)
            result_df = get_preds_kfold(model_path, data_path, conf.run_type)
            metric_df = eval_model(result_df)

            result_df.write_csv(
                out_path / f"predictions_{model_name}_{data_name}_{model_type}.csv"
            )

            metric_df.write_csv(
                out_path / f"metrics_{model_name}_{data_name}_{model_type}.csv"
            )


@hydra.main(config_path="../configs/", version_base="1.3", config_name="eval")
def main(conf: DictConfig) -> None:
    """The main entry point of the script.

    Args:
        conf (DictConfig): The configuration object, passed in by Hydra.

    Returns:
        None
    """
    print(conf)
    get_results(conf)


if __name__ == "__main__":
    main()
