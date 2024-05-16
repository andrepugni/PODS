import pandas as pd
import numpy as np
import os
from statsmodels.formula.api import ols
from src.utils import get_rdd_robust_results_reduced, set_seed


def placebo_test_outcome():
    """
    Compute the ATT for the placebo outcome
    """
    db = pd.DataFrame()
    res = pd.DataFrame()
    target_coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.70, 0.80, 0.90]
    epochs_dict = {
        "cifar10h": 150,
        "galaxyzoo": 50,
        "chestxray2": 3,
        "hatespeech": 100,
        "synth": 50,
    }
    set_seed(42)
    for data in ["cifar10h", "galaxyzoo", "chestxray2", "hatespeech", "synth"]:
        for method in ["RS", "CC", "DT", "LCE", "SP", "OVA", "ASM"]:
            if data == "synth":
                filename_cal = "resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_0.1_0.3_0.2_ep{}.csv".format(
                    data, data, method, epochs_dict[data]
                )
                filename_test = "resultsRAW/{}/GCresultsRAW_test_{}_{}_42_0.1_0.3_0.2_ep{}.csv".format(
                    data, data, method, epochs_dict[data]
                )
            else:
                filename_cal = (
                    "resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_ep{}.csv".format(
                        data, data, method, epochs_dict[data]
                    )
                )
                filename_test = (
                    "resultsRAW/{}/GCresultsRAW_test_{}_{}_42_ep{}.csv".format(
                        data, data, method, epochs_dict[data]
                    )
                )
            if os.path.exists(filename_cal) and os.path.exists(filename_test):
                df_cal = pd.read_csv(filename_cal)
                df_test = pd.read_csv(filename_test)
            else:
                raise FileNotFoundError(
                    "Run train.py for the method {} and data {} first!".format(
                        method, data
                    )
                )
            human_correct = np.where(
                df_test["hum_preds"] == df_test["labels"], 1, 0
            ).astype(float)
            ML_correct = np.where(df_test["preds"] == df_test["labels"], 1, 0).astype(
                float
            )
            set_seed(42)
            correct = np.random.choice([0, 1], size=len(human_correct))
            for c in target_coverages:
                cutoff = np.quantile(df_cal["rej_score"], c)
                defer = np.where(df_test["rej_score"] >= cutoff, 1, 0).astype(float)
                final_pred = np.where(
                    df_test["rej_score"] >= cutoff,
                    df_test["hum_preds"],
                    df_test["preds"],
                )
                # ATT = np.mean(human_correct[defer==1])-np.mean(ML_correct[defer==1])
                # check = human_correct[defer==1]-ML_correct[defer==1]

                tmp_res = pd.DataFrame()
                tmp_res["data"] = [data if data != "chestxray2" else "xray-airspace"]
                tmp_res["method"] = [method]
                tmp_res["target_coverage"] = [c]
                try:
                    df_rdd, _ = get_rdd_robust_results_reduced(
                        correct, df_test["rej_score"], cutoff=cutoff
                    )
                    tmp_res = pd.concat([tmp_res, df_rdd], axis=1)
                except:
                    df_rdd = pd.DataFrame()
                    value_input = 0
                    df_rdd["coef_rob"] = value_input
                    df_rdd["se_rob"] = value_input
                    df_rdd["pv_rob"] = value_input
                    df_rdd["ci_rob_l"] = value_input
                    df_rdd["ci_rob_l"] = value_input
                    df_rdd["ci_rob_u"] = value_input
                    tmp_res = pd.concat([tmp_res, df_rdd], axis=1)
                res = pd.concat([res, tmp_res])
    res.to_csv("results/all_resultsPLACEBO.csv", index=False)


def placebo_test_cutofflow():
    """
    Compute the ATT for the placebo cutoff low
    """
    res = pd.DataFrame()
    target_coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.70, 0.80, 0.90]
    epochs_dict = {
        "cifar10h": 150,
        "galaxyzoo": 50,
        "chestxray2": 3,
        "hatespeech": 100,
        "synth": 50,
    }
    for data in ["cifar10h", "galaxyzoo", "chestxray2", "hatespeech", "synth"]:
        for method in ["RS", "CC", "DT", "LCE", "SP", "OVA", "ASM"]:
            if data == "synth":
                filename_cal = "resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_0.1_0.3_0.2_ep{}.csv".format(
                    data, data, method, epochs_dict[data]
                )
                filename_test = "resultsRAW/{}/GCresultsRAW_test_{}_{}_42_0.1_0.3_0.2_ep{}.csv".format(
                    data, data, method, epochs_dict[data]
                )
            else:
                filename_cal = (
                    "resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_ep{}.csv".format(
                        data, data, method, epochs_dict[data]
                    )
                )
                filename_test = (
                    "resultsRAW/{}/GCresultsRAW_test_{}_{}_42_ep{}.csv".format(
                        data, data, method, epochs_dict[data]
                    )
                )
            if os.path.exists(filename_cal) and os.path.exists(filename_test):
                df_cal = pd.read_csv(filename_cal)
                df_test = pd.read_csv(filename_test)
            else:
                raise FileNotFoundError(
                    "Run train.py for the method {} and data {} first!".format(
                        method, data
                    )
                )
            human_correct = np.where(
                df_test["hum_preds"] == df_test["labels"], 1, 0
            ).astype(float)
            ML_correct = np.where(df_test["preds"] == df_test["labels"], 1, 0).astype(
                float
            )
            for c in target_coverages:
                cutoff = np.quantile(df_cal["rej_score"], c)
                cutoff_low = np.quantile(
                    df_cal[df_cal["rej_score"] <= cutoff]["rej_score"].values, 0.75
                )
                defer = np.where(df_test["rej_score"] >= cutoff, 1, 0).astype(float)
                fake_defer = np.where(df_test["rej_score"] >= cutoff_low, 1, 0).astype(
                    float
                )
                final_pred = np.where(
                    df_test["rej_score"] >= cutoff,
                    df_test["hum_preds"],
                    df_test["preds"],
                )
                correct = np.where(final_pred == df_test["labels"], 1, 0).astype(float)
                ATT = np.mean(human_correct[defer == 1]) - np.mean(
                    ML_correct[defer == 1]
                )
                check = human_correct[defer == 1] - ML_correct[defer == 1]
                tmp = pd.DataFrame()
                tmp["check"] = check
                tmp["defer"] = 1
                dd = ols("check ~ 1", data=tmp).fit(cov_type="HC3")
                acc_hum = np.mean(human_correct[fake_defer == 1])
                acc_ML = np.mean(ML_correct[fake_defer == 0])
                acc_system = np.mean(correct)
                tmp_res = pd.DataFrame()
                tmp_res["data"] = [data if data != "chestxray2" else "xray-airspace"]
                tmp_res["method"] = [method]
                tmp_res["target_coverage"] = [c]
                tmp_res["ATT"] = [ATT]
                tmp_res["ci_l_ATT"] = [dd.conf_int().iloc[:, 0].values[0]]
                tmp_res["ci_u_ATT"] = [dd.conf_int().iloc[:, 1].values[0]]
                tmp_res["pv_rob_ATT"] = [dd.pvalues[0]]
                tmp_res["acc_hum"] = [acc_hum]
                tmp_res["acc_ML"] = [acc_ML]
                tmp_res["acc_system"] = [acc_system]
                try:
                    df_rdd, _ = get_rdd_robust_results_reduced(
                        correct, df_test["rej_score"], cutoff=cutoff_low
                    )
                    tmp_res = pd.concat([tmp_res, df_rdd], axis=1)
                except:
                    df_rdd = pd.DataFrame()
                    value_input = 0
                    df_rdd["coef_rob"] = value_input
                    df_rdd["se_rob"] = value_input
                    df_rdd["pv_rob"] = value_input
                    df_rdd["ci_rob_l"] = value_input
                    df_rdd["ci_rob_l"] = value_input
                    df_rdd["ci_rob_u"] = value_input
                    tmp_res = pd.concat([tmp_res, df_rdd], axis=1)
                res = pd.concat([res, tmp_res])
    res.to_csv("results/all_resultsPLACEBOCUTOFFS_LOW.csv", index=False)


def placebo_test_cutoffhigh():
    """
    Compute the ATT for the placebo cutoff high
    """
    res = pd.DataFrame()
    target_coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.70, 0.80, 0.90]
    epochs_dict = {
        "cifar10h": 150,
        "galaxyzoo": 50,
        "chestxray2": 3,
        "hatespeech": 100,
        "synth": 50,
    }
    for data in ["cifar10h", "galaxyzoo", "chestxray2", "hatespeech", "synth"]:
        for method in ["RS", "CC", "DT", "LCE", "SP", "OVA", "ASM"]:
            if data == "synth":
                filename_cal = "resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_0.1_0.3_0.2_ep{}.csv".format(
                    data, data, method, epochs_dict[data]
                )
                filename_test = "resultsRAW/{}/GCresultsRAW_test_{}_{}_42_0.1_0.3_0.2_ep{}.csv".format(
                    data, data, method, epochs_dict[data]
                )
            else:
                filename_cal = (
                    "resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_ep{}.csv".format(
                        data, data, method, epochs_dict[data]
                    )
                )
                filename_test = (
                    "resultsRAW/{}/GCresultsRAW_test_{}_{}_42_ep{}.csv".format(
                        data, data, method, epochs_dict[data]
                    )
                )
            if os.path.exists(filename_cal) and os.path.exists(filename_test):
                df_cal = pd.read_csv(filename_cal)
                df_test = pd.read_csv(filename_test)
            else:
                raise FileNotFoundError(
                    "Run train.py for the method {} and data {} first!".format(
                        method, data
                    )
                )
            human_correct = np.where(
                df_test["hum_preds"] == df_test["labels"], 1, 0
            ).astype(float)
            ML_correct = np.where(df_test["preds"] == df_test["labels"], 1, 0).astype(
                float
            )
            for c in target_coverages:
                cutoff = np.quantile(df_cal["rej_score"], c)
                cutoff_high = np.quantile(
                    df_cal[df_cal["rej_score"] >= cutoff]["rej_score"], 0.25
                )
                defer = np.where(df_test["rej_score"] >= cutoff, 1, 0).astype(float)
                fake_defer = np.where(df_test["rej_score"] >= cutoff_high, 1, 0).astype(
                    float
                )
                final_pred = np.where(
                    df_test["rej_score"] >= cutoff,
                    df_test["hum_preds"],
                    df_test["preds"],
                )
                correct = np.where(final_pred == df_test["labels"], 1, 0).astype(float)
                ATT = np.mean(human_correct[defer == 1]) - np.mean(
                    ML_correct[defer == 1]
                )
                check = human_correct[defer == 1] - ML_correct[defer == 1]
                tmp = pd.DataFrame()
                tmp["check"] = check
                tmp["defer"] = 1
                dd = ols("check ~ 1", data=tmp).fit(cov_type="HC3")
                acc_hum = np.mean(human_correct[fake_defer == 1])
                acc_ML = np.mean(ML_correct[fake_defer == 0])
                acc_system = np.mean(correct)
                tmp_res = pd.DataFrame()
                tmp_res["data"] = [data if data != "chestxray2" else "xray-airspace"]
                tmp_res["method"] = [method]
                tmp_res["target_coverage"] = [c]
                tmp_res["ATT"] = [ATT]
                tmp_res["ci_l_ATT"] = [dd.conf_int().iloc[:, 0].values[0]]
                tmp_res["ci_u_ATT"] = [dd.conf_int().iloc[:, 1].values[0]]
                tmp_res["pv_rob_ATT"] = [dd.pvalues[0]]
                tmp_res["acc_hum"] = [acc_hum]
                tmp_res["acc_ML"] = [acc_ML]
                tmp_res["acc_system"] = [acc_system]
                try:
                    df_rdd, _ = get_rdd_robust_results_reduced(
                        correct, df_test["rej_score"], cutoff=cutoff_high
                    )
                    tmp_res = pd.concat([tmp_res, df_rdd], axis=1)
                except:
                    df_rdd = pd.DataFrame()
                    value_input = 0
                    df_rdd["coef_rob"] = value_input
                    df_rdd["se_rob"] = value_input
                    df_rdd["pv_rob"] = value_input
                    df_rdd["ci_rob_l"] = value_input
                    df_rdd["ci_rob_l"] = value_input
                    df_rdd["ci_rob_u"] = value_input
                    tmp_res = pd.concat([tmp_res, df_rdd], axis=1)
                res = pd.concat([res, tmp_res])
    res.to_csv("results/all_resultsPLACEBOCUTOFFS_HIGH.csv", index=False)


def main():
    placebo_test_outcome()
    placebo_test_cutofflow()
    placebo_test_cutoffhigh()


if __name__ == "__main__":
    main()
