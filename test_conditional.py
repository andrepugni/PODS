import pandas as pd
import numpy as np
import os
from statsmodels.formula.api import ols
from src.utils import get_rdd_robust_results_reduced


def main():
    res = pd.DataFrame()
    target_coverages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.70, 0.80, 0.90]
    epochs_dict = {
        "cifar10h": 150,
        "galaxyzoo": 50,
        "chestxray2": 3,
        "hatespeech": 100,
        "synth": 50,
    }
    # collect all results
    for data in ["chestxray2"]:
        # for each method compute effects of interest
        for method in ["RS", "CC", "DT", "LCE", "SP", "OVA", "ASM"]:
            filename_cal = "./resultsRAW/{}/GCresultsRAW_cal_{}_{}_42_ep{}.csv".format(
                data, data, method, epochs_dict[data]
            )
            filename_test = "./resultsRAW/chestxray2/{}_Scenario1_Conditional_test_chestxray{}_42_ep3.csv".format(
                method, 2
            )
            if os.path.exists(filename_cal) and os.path.exists(filename_test):
                df_cal = pd.read_csv(filename_cal)
                df_test = pd.read_csv(filename_test)
            else:
                import pdb

                pdb.set_trace()
                raise FileNotFoundError(
                    "Run train.py for the method {} and data {} first!".format(
                        method, data
                    )
                )
            filename_test = "./resultsRAW/chestxray2/RS_Scenario1_Conditional_test_chestxray2_42_ep3.csv"
            # compute correct predictions by the human
            human_correct = np.where(
                df_test["hum_preds"] == df_test["labels"], 1, 0
            ).astype(float)
            # compute correct predictions by the ML model
            ML_correct = np.where(df_test["preds"] == df_test["labels"], 1, 0).astype(
                float
            )
            for (
                c
            ) in (
                target_coverages
            ):  # for each target coverage compute the tau_ATT and tau_RDD
                if c == 0:
                    cutoff = -100
                    defer = np.ones(len(df_test))
                else:
                    cutoff = np.quantile(df_cal["rej_score"], c)
                    defer = np.where(df_test["rej_score"] >= cutoff, 1, 0).astype(float)
                final_pred = np.where(
                    df_test["rej_score"] >= cutoff,
                    df_test["hum_preds"],
                    df_test["preds"],
                )
                correct = np.where(final_pred == df_test["labels"], 1, 0).astype(float)
                ATT = np.mean(human_correct[defer == 1]) - np.mean(
                    ML_correct[defer == 1]
                )
                filter_male = df_test["Patient Gender"] == "M"
                filter_female = df_test["Patient Gender"] == "F"
                cate_dict = {"Male": 0, "Female": 0}
                filters = [filter_male, filter_female]
                dict_filters = {k: v for k, v in zip(cate_dict.keys(), filters)}
                tmp_res = pd.DataFrame()
                tmp_res["data"] = [data if data != "chestxray2" else "xray-airspace"]
                tmp_res["method"] = [method]
                tmp_res["target_coverage"] = [c]
                check = human_correct - ML_correct
                tmp = pd.DataFrame()
                tmp["check"] = check
                tmp["defer"] = defer
                tmp2 = df_test.copy()
                filter_male = tmp2["Patient Gender"] == "M"
                filter_female = tmp2["Patient Gender"] == "F"
                tmp["Male"] = np.where((filter_male), 1, 0)
                tmp["Female"] = np.where((filter_female), 1, 0)
                # CATE for Gender
                dd = ols("check ~ -1 + Male + Female", data=tmp[tmp["defer"] == 1]).fit(
                    cov_type="HC1"
                )
                # here we attach coefs
                coefs_to_attach = pd.DataFrame(
                    dd.params.values.reshape(-1, 1).T,
                    columns=["CATE_{}".format(el) for el in dd.pvalues.index.values],
                )
                # here we attach ci_low
                ci_low_to_attach = pd.DataFrame(
                    dd.conf_int().iloc[:, 0].values.reshape(-1, 1).T,
                    columns=[
                        "CATE_ci_low_{}".format(el)
                        for el in dd.conf_int().iloc[:, 0].index.values
                    ],
                )
                # here we attach ci_high
                ci_high_to_attach = pd.DataFrame(
                    dd.conf_int().iloc[:, 1].values.reshape(-1, 1).T,
                    columns=[
                        "CATE_ci_high_{}".format(el)
                        for el in dd.conf_int().iloc[:, 1].index.values
                    ],
                )
                # here we attach pvalues
                pv_to_attach = pd.DataFrame(
                    dd.pvalues.values.reshape(-1, 1).T,
                    columns=["pv_rob_{}".format(el) for el in dd.pvalues.index.values],
                )

                count_males = np.sum(tmp[tmp["defer"] == 1]["Male"])
                count_females = np.sum(tmp[tmp["defer"] == 1]["Female"])
                acc_hum = np.mean(human_correct[defer == 1])
                # acc_ML = np.mean(ML_correct[defer == 0])
                acc_system = np.mean(correct)
                tmp_res_ = pd.DataFrame()
                tmp_res = pd.concat(
                    [
                        tmp_res,
                        coefs_to_attach,
                        ci_low_to_attach,
                        ci_high_to_attach,
                        pv_to_attach,
                    ],
                    axis=1,
                )
                tmp_res["acc_hum"] = [acc_hum]
                tmp_res["count_female"] = [count_females]
                tmp_res["count_males"] = [count_males]
                # tmp_res["acc_ML"] = [acc_ML]
                tmp_res["acc_system"] = [acc_system]
                res = pd.concat([res, tmp_res], axis=0)
            # compute measures for target coverage 1 and 0
    res.to_csv("./results/all_results_cond.csv", index=False)  # save results


if __name__ == "__main__":
    main()
