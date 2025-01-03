# A utility file containing functions to read and preprocess data
# and print statistics about the data

import polars as pl
import polars.selectors as cs
from pathlib import Path
from typing import Union, Tuple, Any
from numpy import ndarray


def get_data(
    data_path: Union[Path, str], run_type: str, return_as_Xy: bool = False
) -> Union[pl.DataFrame, Tuple[ndarray[Any, Any], ndarray[Any, Any]]]:
    """Loads the data from a given path and returns it as a DataFrame.

    Parameters
    ----------
    data_path : Union[Path, str]
        Path to the data file.

    run_type: str
        One of `full`, `geno_only`, `chem_only`

    return_as_Xy: bool
        If you want the data to be returned as [Features, Observations]

    Returns
    -------
    Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]
        The loaded data as a DataFrame.
    """

    if isinstance(data_path, Path):
        format = data_path.suffix
    elif isinstance(data_path, str):
        format = data_path.split(".")[-1]
    else:
        raise ValueError(
            f"path must be either Path or str but it is of type {type(data_path)}"
        )

    match format:
        case "feather":
            df = pl.read_ipc(data_path)
        case "parquet":
            df = pl.read_parquet(data_path)
        case "csv":
            df = pl.read_csv(data_path)
        case _:
            raise NotImplementedError(
                "File format not supported yet. Most be one of 'feather', 'parquet', 'csv'"
            )

    if not return_as_Xy:
        # y = df['Phenotype'].to_numpy()

        match run_type:
            case "geno_only":
                df = df.select(
                    pl.col("Strain"),
                    cs.starts_with(
                        "Y"
                    ),  # Only the genotype columns, based on yeast systemic names # TODO: Hardcoded for now. But should be flexible
                    pl.col("Condition"),
                    pl.col("Phenotype"),
                )
                return df

            case "chem_only":
                df = df.select(
                    pl.col("Strain"),
                    cs.contains(
                        "latent"
                    ),  # Only the chemical information # TODO: Hardcoded for now. But should be flexible
                    pl.col("Condition"),
                    pl.col("Phenotype"),
                )
                return df
            case "full":
                return df
            case _:
                raise ValueError(
                    "Invalid run_type. Must be one of ['geno_only', 'chem_only', 'full']"
                )

    else:
        y = df["Phenotype"].to_numpy()

        match run_type:
            case "geno_only":
                X = df.select(cs.starts_with("Y")).to_numpy()
                return X, y
            case "chem_only":
                X = df.select(cs.contains("latent")).to_numpy()
                return X, y
            case "full":
                X = df.select(cs.starts_with("Y"), cs.contains("latent")).to_numpy()
                return X, y
            case _:
                raise ValueError(
                    "Invalid run_type. Must be one of ['geno_only', 'chem_only', 'full']"
                )


def get_data_stats(data_path: Union[Path, str], run_type: str = "full") -> None:
    """Prints statistics about the data.

    Parameters
    ----------
    data_path : Union[Path, str]
        Path to the data file.
    run_type: str
        One of `full`, `geno_only`, `chem_only`
    """
    df = get_data(data_path=data_path, run_type=run_type)
    print(f"Shape of the data: {df.shape}")
    print(f"Number of unique Strains: \n {df['Strain'].n_unique()}")
    print(f"Number of unique Conditions: \n {df['Condition'].value_counts()}")
    print(f"Number of unique Phenotypes: \n {df['Phenotype'].value_counts()}")

    # A little more involved: identifying how many genes have some variation

    gene_df = df.select("Strain", cs.starts_with("Y")).unique()
    gene_df = gene_df.melt(
        id_vars="Strain", variable_name="Gene", value_name="Mutation"
    )
    gene_df = gene_df.group_by("Gene").agg(pl.col("Mutation").var().alias("Variation"))

    print(gene_df.filter(pl.col("Variation") > 0))


if __name__ == "__main__":
    # Small application to get some statistics about the data

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument("--run_type", type=str, default="full")
    args = parser.parse_args()

    get_data_stats(data_path=args.data_path, run_type=args.run_type)
