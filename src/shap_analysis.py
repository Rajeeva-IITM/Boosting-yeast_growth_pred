from pathlib import Path
from dotenv import load_dotenv
import hydra
import polars as pl
from rich import print
from omegaconf import DictConfig
from utils import get_model, get_model_paths

load_dotenv()


def get_shap_values(
    conf: DictConfig, model_path: Path, data_path: Path, fold: int
) -> pl.DataFrame:
    """Compute SHAP values for a given model and dataset.

    Parameters
    ----------
    model_path : Path
        Path to the pickle file containing the model.
    data_path : Path
        Path to the csv file containing the data.
    fold : int
        The number of the fold to compute SHAP values for.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the mean absolute SHAP values for each feature,
        sorted in descending order.
    """
    model = get_model(model_path)
    data = pl.read_ipc(data_path)
    X = data.drop(["Condition", "Strain", "Phenotype"]).to_pandas()
    y = data["Phenotype"].to_numpy()

    # instantiate explainer
    explainer = hydra.utils.instantiate(conf.explainer, model, X)

    # compute shap values
    shap_values = explainer.shap_values(X, y)
    shap_df = pl.DataFrame(data=shap_values, schema=X.columns)
    shap_mean = (
        shap_df.with_columns(pl.all().abs().mean())
        .unique()
        .melt()
        .with_columns(Fold=pl.lit(fold))
        .rename({"value": "Value", "variable": "Feature"}, axis=1)
        .sort("Value", descending=True)
    )

    return shap_mean


def get_shap_folds(
    conf: DictConfig, model_paths: Path, data_path: Path
) -> pl.DataFrame:
    """Compute SHAP values for multiple models from the same dataset and concatenate them. Useful
    for when when you have models trained on different folds of the underlying data.

    Parameters
    ----------
    conf : DictConfig
        Configuration object containing the paths to the models and the data.
    model_paths : Path
        Paths to the pickle files containing the models.
    data_path : Path
        Path to the csv file containing the data.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the mean absolute SHAP values for each feature,
        sorted in descending order, computed for each model and concatenated.
    """
    df_folds = []
    for fold, model_path in enumerate(model_paths):
        df_fold = get_shap_values(conf, model_path, data_path, fold)
        df_folds.append(df_fold)

    final_df = pl.concat(df_folds)
    final_df = final_df.group_by("Feature", maintain_order=True).agg(
        pl.col("Value").mean()
    )
    return final_df


@hydra.main(config_path="../configs/", version_base="1.3", config_name="interpret")
def main(conf: DictConfig) -> None:
    """Main entry point for the script.

    The script takes in a configuration, reads the models and data from the configuration,
    computes the SHAP values for each model, and saves the results to a CSV file.

    Parameters
    ----------
    conf : DictConfig
        The configuration object, passed in by Hydra.

    Returns
    -------
    None
    """

    print(conf)

    out_path = Path(conf.out_path)
    model_type = conf.model_load_keys.model_type
    model_paths = get_model_paths(**conf.model_load_keys)

    assert (
        len(conf.model_load_keys.model_names) <= len(conf.data_paths)
    ), f"Mismatch between data {len(conf.data_paths)} and model {len(conf.model_load_keys.model_names)} and model names"

    print(model_paths)

    for model_name, model_path in model_paths.items():
        shap_df = get_shap_folds(conf, model_path, conf.data_paths.get(model_name))
        shap_df.write_csv(out_path / f"shap_{model_name}_{model_type}.csv")


if __name__ == "__main__":
    main()
    # run_path = Path("/home/rajeeva/Project/runs/chem_subsets/Final/")
    # data_path = Path("/home/rajeeva/Project/data/chem_subsets/new_latent/")
    # out_path = Path("/home/rajeeva/Project/boosting/results/shap/")

    # # model_paths = {
    # #     "Bloom2013": run_path / "Carbons_Bloom2013_Modified_hyperparams/Boosting.pkl",
    # #     "Bloom2013_binary": run_path
    # #     / "Carbons_Bloom2013_Modified_hyperparams_binary/Boosting.pkl",
    # #     "Bloom2015": run_path / "Carbons_Bloom2015_Modified_hyperparams/Boosting.pkl",
    # #     "Bloom2015_binary": run_path
    # #     / "Carbons_Bloom2015_Modified_hyperparams_binary/Boosting.pkl",
    # #     "Bloom2019": run_path / "Carbons_Bloom2019_Modified_hyperparams/Boosting.pkl",
    # #     "Bloom2019_binary": run_path
    # #     / "Carbons_Bloom2019_Modified_hyperparams_binary/Boosting.pkl",
    # #     "Bloom2019_BYxM22": run_path
    # #     / "Carbons_Bloom2019_BYxM22_Modified_hyperparams/Boosting.pkl",
    # #     "Bloom2019_BYxM22_binary": run_path
    # #     / "Carbons_Bloom2019_BYxM22_Modified_hyperparams_binary/Boosting.pkl",
    # #     "Bloom2019_RMxYPS163": run_path
    # #     / "Carbons_Bloom2019_RMxYPS163_Modified_hyperparams/Boosting.pkl",
    # #     "Bloom2019_RMxYPS163_binary": run_path
    # #     / "Carbons_Bloom2019_RMxYPS163_Modified_hyperparams_binary/Boosting.pkl",
    # # }

    # model_paths = {
    #     "Bloom2013": list(run_path.glob("Carbons_Bloom2013*/Boosting.pkl")),
    #     "Bloom2013_RF": list(run_path.glob("Carbons_Bloom2013*/RandomForest.pkl")),
    #     # "Bloom2013_binary": run_path.glob("Carbons_Bloom2013_Modified_hyperparams_binary_*.pkl"),
    #     "Bloom2015": list(run_path.glob("Carbons_Bloom2015*/Boosting.pkl")),
    #     "Bloom2015_RF": list(run_path.glob("Carbons_Bloom2015*/RandomForest.pkl")),
    #     # "Bloom2015_binary": run_path.glob("Carbons_Bloom2015_Modified_hyperparams_binary_*.pkl"),
    #     "Bloom2019": list(run_path.glob("Carbons_Bloom2019*/Boosting.pkl")),
    #     "Bloom2019_RF": list(run_path.glob("Carbons_Bloom2019*/RandomForest.pkl")),
    #     # "Bloom2019_binary": run_path.glob("Carbons_Bloom2019_Modified_hyperparams_binary_*.pkl"),
    #     "Bloom2019_BYxM22": list(
    #         run_path.glob("Carbons_Bloom2019_BYxM22_*/Boosting.pkl")
    #     ),
    #     "Bloom2019_BYxM22_RF": list(
    #         run_path.glob("Carbons_Bloom2019_BYxM22_*/RandomForest.pkl")
    #     ),
    #     # "Bloom2019_BYxM22_binary": run_path.glob("Carbons_Bloom2019_BYxM22_Modified_hyperparams_binary_*.pkl"),
    #     "Bloom2019_RMxYPS163": list(
    #         run_path.glob("Carbons_Bloom2019_RMxYPS163_*/Boosting.pkl")
    #     ),
    #     "Bloom2019_RMxYPS163_RF": list(
    #         run_path.glob("Carbons_Bloom2019_RMxYPS163_*/RandomForest.pkl")
    #     ),
    #     # "Bloom2019_RMxYPS163_binary": run_path.glob("Carbons_Bloom2019_RMxYPS163_Modified_hyperparams_binary_*.pkl"),
    # }

    # data_paths = {
    #     "Bloom2013": data_path / "carbons/carbons_bloom2013_clf_3_pubchem.feather",
    #     "Bloom2015": data_path / "carbons/carbons_bloom2015_clf_3_pubchem.feather",
    #     "Bloom2019": data_path / "carbons/carbons_bloom2019_clf_3_pubchem.feather",
    #     "Bloom2019_BYxM22": data_path
    #     / "carbons/carbons_bloom2019_BYxM22_clf_3_pubchem.feather",
    #     "Bloom2019_RMxYPS163": data_path
    #     / "carbons/carbons_bloom2019_RMxYPS163_clf_3_pubchem.feather",
    #     "Bloom2013_RF": data_path / "carbons/carbons_bloom2013_clf_3_pubchem.feather",
    #     "Bloom2015_RF": data_path / "carbons/carbons_bloom2015_clf_3_pubchem.feather",
    #     "Bloom2019_RF": data_path / "carbons/carbons_bloom2019_clf_3_pubchem.feather",
    #     "Bloom2019_BYxM22_RF": data_path
    #     / "carbons/carbons_bloom2019_BYxM22_clf_3_pubchem.feather",
    #     "Bloom2019_RMxYPS163_RF": data_path
    #     / "carbons/carbons_bloom2019_RMxYPS163_clf_3_pubchem.feather",
    #     # "Bloom2019_binary": data_path
    #     # / "binary/carbons/carbons_bloom2019_clf_3_pubchem.feather",
    #     # "Bloom2013_binary": data_path
    #     # / "binary/carbons/carbons_bloom2013_clf_3_pubchem.feather",
    #     # "Bloom2015_binary": data_path
    #     # / "binary/carbons/carbons_bloom2015_clf_3_pubchem.feather",
    #     # "Bloom2019_BYxM22_binary": data_path
    #     # / "binary/carbons/carbons_bloom2019_BYxM22_clf_3_pubchem.feather",
    #     # "Bloom2019_RMxYPS163_binary": data_path
    #     # / "binary/carbons/carbons_bloom2019_RMxYPS163_clf_3_pubchem.feather",
    # }

    # for name, model_path in model_paths.items():
    #     print(name)
    #     result = get_shap_folds(model_path, data_paths[name])
    #     result.write_csv(out_path.joinpath(f"{name}.csv"))
