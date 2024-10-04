from src.datasets import ChestXrayDataset
from src.utils import set_seed
from torch import optim
import pandas as pd
import numpy as np


def main():
    """
    The script adds the demographic info to the test results of the ChestXray dataset.
    """
    seed = 42
    label_chosen = 2
    set_seed(seed)
    # load the dataset
    dataset = ChestXrayDataset(
        False,
        True,
        data_dir="data",
        label_chosen=label_chosen,
        batch_size=128,
        test_split=0.2,
        val_split=0.10,
    )
    # load the demographic info
    db = pd.read_csv("data/Data_Entry_2017_v2020.csv")
    # merge the demographic info with the test results
    test_data = pd.DataFrame(dataset.data_test_loader.dataset.images, columns=["Index"])
    test_data["Image Index"] = test_data["Index"].str.split("/", expand=True)[2].copy()
    df = test_data.merge(db, on="Image Index")
    # save the results
    for method in ["ASM", "SP", "RS", "CC", "OVA", "DT", "LCE"]:
        raws = pd.read_csv(
            "resultsRAW/chestxray2/GCresultsRAW_test_chestxray2_{}_42_ep3.csv".format(
                method
            )
        )

        tmp = pd.concat([df, raws], axis=1)
        cols = [
            "Image Index",
            "Patient ID",
            "Patient Age",
            "Patient Gender",
            "rej_score",
            "labels",
            "hum_preds",
            "preds",
            "class_probs_0",
            "class_probs_1",
        ]
        tmp = tmp[cols].copy()
        tmp.to_csv(
            "resultsRAW/chestxray2/{}_Scenario1_Conditional_test_chestxray{}_42_ep3.csv".format(
                method, label_chosen
            ),
            index=False,
        )


if __name__ == "__main__":
    main()
