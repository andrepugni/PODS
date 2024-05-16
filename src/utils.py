"""
The code is adapted from the following repository:
https://github.com/clinicalml/human_ai_deferral
"""


import torch
import random
import urllib.request
import os
import tarfile
import logging
from tqdm import tqdm
import sklearn.metrics
import pandas as pd
from rdrobust import rdrobust
from torch.utils.data import Dataset
import torch.nn.functional as F
from src.metrics import *

def set_seed(seed=None, seed_torch=True):
    """
    Set random seed for reproducibility.
    :param seed: the random seed. If None, no action is taken. Default: None
    :param seed_torch: if True, sets the random seed for pytorch. Default: True.
    :return:
    """
    if seed is None:
        seed = np.random.choice(2**32)
        random.seed(seed)
        np.random.seed(seed)
    if seed_torch:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(f"Random seed {seed} has been set.")


def download_images_NIH(data_dir="data/"):
    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    if os.path.exists("data/") is False:
        os.mkdir("data/") # create the data directory if it does not exist
    if os.path.exists("data/{}".format(data_dir)) is False:
        os.mkdir("data/{}".format(data_dir)) # create the data directory if it does not exist

    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1) # filename
        print('downloading'+fn+'...') # print out the filename
        urllib.request.urlretrieve(link, "data/{}/{}".format(data_dir, fn))  # download the zip file

    print("Download complete. Please check the checksums")


def train_deferral_single_model(Method, dataloader_train, dataloader_test, epochs, lr, verbose=True, test_interval=5,
                                include_scheduler=False):
    optimizer = torch.optim.SGD(Method.model.parameters(), lr,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader_train) * epochs)
    for epoch in tqdm(range(epochs)):
        Method.fit_epoch(dataloader_train, optimizer, verbose, epoch)
        if verbose and epoch % test_interval == 0:
            data_test = Method.test(dataloader_test)
            print(compute_deferral_metrics(data_test))
        if include_scheduler:
            scheduler.step()

    final_test = Method.test(dataloader_test)
    return compute_deferral_metrics(final_test)


def train_single_model(Method, model, fit, dataloader_train, dataloader_test, epochs, verbose=True, test_interval=5):
    '''
    Method: the method class
    model: model in method
    fit: fit method in Method class
    '''
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader_train) * epochs)
    for epoch in tqdm(range(epochs)):
        Method.fit(epoch, dataloader_train, optimizer, verbose, epoch)
        if epoch % test_interval == 0:
            data_test = Method.test(dataloader_test)
            print(compute_classification_metrics(data_test))
        scheduler.step()
    final_test = Method.test(dataloader_test)
    return compute_classification_metrics(final_test)

def weighted_cross_entropy_loss(outputs, labels, weights, weight_class=None):
    """
    Weigthed cross entropy loss
    outputs: network outputs with softmax
    labels: target
    weights: weights for each example

    return: weighted cross entropy loss as scalar
    """
    outputs = weights * F.cross_entropy(outputs, labels, reduction="none", weight=weight_class)  # regular CE
    return torch.sum(outputs) / (torch.sum(weights)+1e-8)

def compute_deferral_metrics(data_test):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels', 'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    results["human_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"], data_test["labels"]
    )
    results["coverage"] = 1 - np.mean(data_test["defers"])
    # get classifier accuracy when defers is 0
    results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"][data_test["defers"] == 0],
        data_test["labels"][data_test["defers"] == 0],
    )
    # get human accuracy when defers is 1
    results["human_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"][data_test["defers"] == 1],
        data_test["labels"][data_test["defers"] == 1],
    )
    # get system accuracy
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (1 - data_test["defers"])
        + data_test["hum_preds"] * (data_test["defers"]),
        data_test["labels"],
    )
    return results


def compute_classification_metrics(data_test):
    """compute metrics for just classification

    Args:
        data_test (dict): dict data with fields 'labels',  'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data, also returns AUC for preds_proba
    """

    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    # check if preds and labels are binary
    if (
        len(np.unique(data_test["labels"])) == 2
        and len(np.unique(data_test["preds"])) == 2
    ):
        # get f1
        results["classifier_all_f1"] = sklearn.metrics.f1_score(
            data_test["preds"], data_test["labels"]
        )
        if "preds_proba" in data_test:
            results["auc"] = sklearn.metrics.roc_auc_score(
                data_test["labels"], data_test["preds_proba"]
            )
        else:
            results["auc"] = sklearn.metrics.roc_auc_score(
                data_test["labels"], data_test["preds"]
            )
    return results


def compute_coverage_v_acc_curve(data_test):
    """

    Args:
        data_test (dict): dict data with field   {'defers': defers_all, 'labels': truths_all, 'hum_preds': hum_preds_all, 'preds': predictions_all, 'rej_score': rej_score_all, 'class_probs': class_probs_all}

    Returns:
        data (list): compute_deferral_metrics(data_test_modified) on different coverage levels, first element of list is compute_deferral_metrics(data_test)
    """
    # get unique rejection scores
    rej_scores = np.unique(data_test["rej_score"])
    # sort by rejection score
    # get the 100 quantiles for rejection scores
    rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    # for each quantile, get the coverage and accuracy by getting a new deferral decision
    all_metrics = []
    all_metrics.append(compute_deferral_metrics(data_test))
    for q in rej_scores_quantiles:
        # get deferral decision
        defers = (data_test["rej_score"] > q).astype(int)
        copy_data = copy.deepcopy(data_test)
        copy_data["defers"] = defers
        # compute metrics
        metrics = compute_deferral_metrics(copy_data)
        all_metrics.append(metrics)
    return all_metrics

def extract_data(data_dir="data/", max_links=12, dataset="nih"):
    if dataset == "nih":
        files_to_extract = [f"nih/images_{i:02d}.tar.gz" for i in range(1, max_links + 1)]
        # extract files
        for f in files_to_extract:
            fn = "{}/{}".format(data_dir, f)
            logging.info("Extracting " + fn + "...")
            # os.system('tar -zxvf '+fn+' -C '+self.data_dir+'/images_nih')
            file = tarfile.open(fn)
            file.extractall(data_dir + "/images_nih")
            file.close()
            logging.info("Done")

def get_rdd_robust_results(correct, conf_scores, cutoff, rdrobust_params={}):
    """
    Get RDD robust results
    :param correct:
    :param test_vals:
    :param cutoff:
    :param rdrobust_params:
    :return:
     df_rdd: pd.DataFrame
    """
    res = rdrobust(y=correct, x=conf_scores, c=cutoff, **rdrobust_params)
    rdd_params = {
        "N": None,  # vector with the sample sizes used to the left and to the right of the cutoff
        "N_h": None,  # vector with the effective sample sizes used to the left and to the right of the cutoff
        "c": None,  # cutoff value
        "p": None,  # order of the polynomial used for estimation of the regression function
        "q": None,  # order of the polynomial used for estimation of the bias of the regression function
        "bws": None,  # matrix containing the bandwidths used
        "tau_cl": None,  # conventional local-polynomial estimate to the left and to the right of the cutoff
        "tau_bc": None,  # bias-corrected local-polynomial estimate to the left and to the right of the cutoff
        "coef": None,  # vector containing conventional and bias-corrected local-polynomial RD estimates
        "se": None,  # vector containing conventional and robust standard errors of the local-polynomial RD estimates
        "bias": None,  # estimated bias for the local-polynomial RD estimator below and above the cutoff
        "beta_p_l": None,  # conventional p-order local-polynomial estimates to the left of the cutoff
        "beta_p_r": None,  # conventional p-order local-polynomial estimates to the right of the cutoff
        "V_cl_l": None,  # conventional variance-covariance matrix estimated below the cutoff
        "V_cl_r": None,  # conventional variance-covariance matrix estimated above the cutoff
        "V_rb_l": None,  # robust variance-covariance matrix estimated below the cutoff
        "V_rb_r": None,  # robust variance-covariance matrix estimated above the cutoff
        "pv": None,
        # vector containing the p-values associated with conventional, bias-corrected, and robust local-polynomial RD estimates
        "ci": None,
        # matrix containing the confidence intervals associated with conventional, bias-corrected, and robust local-polynomial RD estimates
    }
    for param in rdd_params.keys():
        rdd_params[param] = getattr(res, param)
    rdd_params_clean = {}
    rdd_params_clean['N'] = rdd_params['N']
    rdd_params_clean['N_h_l'] = rdd_params['N_h'][0]
    rdd_params_clean['N_h_r'] = rdd_params['N_h'][1]
    rdd_params_clean['c'] = rdd_params['c']
    rdd_params_clean['p'] = rdd_params['p']
    rdd_params_clean['q'] = rdd_params['q']
    rdd_params_clean['bws_l_h'] = rdd_params['bws']['left'].iloc[0]
    rdd_params_clean['bws_l_b'] = rdd_params['bws']['left'].iloc[1]
    rdd_params_clean['bws_r_h'] = rdd_params['bws']['right'].iloc[0]
    rdd_params_clean['bws_r_b'] = rdd_params['bws']['right'].iloc[1]
    rdd_params_clean['tau_cl_l'] = rdd_params['tau_cl'][0]
    rdd_params_clean['tau_cl_r'] = rdd_params['tau_cl'][1]
    rdd_params_clean['tau_bc_l'] = rdd_params['tau_bc'][0]
    rdd_params_clean['tau_bc_r'] = rdd_params['tau_bc'][1]
    rdd_params_clean['coef_conv'] = rdd_params['coef'].values[0][0]
    rdd_params_clean['coef_bias'] = rdd_params['coef'].values[1][0]
    rdd_params_clean['coef_rob'] = rdd_params['coef'].values[2][0]
    rdd_params_clean['se_conv'] = rdd_params['se'].values[0][0]
    rdd_params_clean['se_bias'] = rdd_params['se'].values[1][0]
    rdd_params_clean['se_rob'] = rdd_params['se'].values[2][0]
    rdd_params_clean['bias_l'] = rdd_params['bias'][0]
    rdd_params_clean['bias_r'] = rdd_params['bias'][1]
    # rdd_params_clean['beta_p_l'] = rdd_params['beta_p_l']
    # rdd_params_clean['beta_p_r'] = rdd_params['beta_p_r']
    rdd_params_clean['pv_conv'] = rdd_params['pv'].values[0][0]
    rdd_params_clean['pv_bias'] = rdd_params['pv'].values[1][0]
    rdd_params_clean['pv_rob'] = rdd_params['pv'].values[2][0]
    rdd_params_clean['ci_conv_l'] = rdd_params['ci']['CI Lower'].values[0]
    rdd_params_clean['ci_conv_u'] = rdd_params['ci']['CI Upper'].values[0]
    rdd_params_clean['ci_bias_l'] = rdd_params['ci']['CI Lower'].values[1]
    rdd_params_clean['ci_bias_u'] = rdd_params['ci']['CI Upper'].values[1]
    rdd_params_clean['ci_rob_l'] = rdd_params['ci']['CI Lower'].values[2]
    rdd_params_clean['ci_rob_u'] = rdd_params['ci']['CI Upper'].values[2]
    df_rdd = pd.DataFrame()
    for param in rdd_params_clean.keys():
        df_rdd[param] = [rdd_params_clean[param]]
    return df_rdd, rdd_params_clean

def get_rdd_robust_results_reduced(correct, conf_scores, cutoff, rdrobust_params={}):
    """
    Get RDD robust results
    :param correct:
    :param test_vals:
    :param cutoff:
    :param rdrobust_params:
    :return:
     df_rdd: pd.DataFrame
    """
    res = rdrobust(y=correct, x=conf_scores, c=cutoff, **rdrobust_params)
    rdd_params = {
        "N": None,  # vector with the sample sizes used to the left and to the right of the cutoff
        "N_h": None,  # vector with the effective sample sizes used to the left and to the right of the cutoff
        "c": None,  # cutoff value
        "p": None,  # order of the polynomial used for estimation of the regression function
        "q": None,  # order of the polynomial used for estimation of the bias of the regression function
        "bws": None,  # matrix containing the bandwidths used
        "tau_cl": None,  # conventional local-polynomial estimate to the left and to the right of the cutoff
        "tau_bc": None,  # bias-corrected local-polynomial estimate to the left and to the right of the cutoff
        "coef": None,  # vector containing conventional and bias-corrected local-polynomial RD estimates
        "se": None,  # vector containing conventional and robust standard errors of the local-polynomial RD estimates
        "bias": None,  # estimated bias for the local-polynomial RD estimator below and above the cutoff
        "beta_p_l": None,  # conventional p-order local-polynomial estimates to the left of the cutoff
        "beta_p_r": None,  # conventional p-order local-polynomial estimates to the right of the cutoff
        "V_cl_l": None,  # conventional variance-covariance matrix estimated below the cutoff
        "V_cl_r": None,  # conventional variance-covariance matrix estimated above the cutoff
        "V_rb_l": None,  # robust variance-covariance matrix estimated below the cutoff
        "V_rb_r": None,  # robust variance-covariance matrix estimated above the cutoff
        "pv": None,
        # vector containing the p-values associated with conventional, bias-corrected, and robust local-polynomial RD estimates
        "ci": None,
        # matrix containing the confidence intervals associated with conventional, bias-corrected, and robust local-polynomial RD estimates
    }
    for param in rdd_params.keys():
        rdd_params[param] = getattr(res, param)
    rdd_params_clean = {}
    rdd_params_clean['N'] = rdd_params['N']
    rdd_params_clean['N_h_l'] = rdd_params['N_h'][0]
    rdd_params_clean['N_h_r'] = rdd_params['N_h'][1]
    rdd_params_clean['c'] = rdd_params['c']
    rdd_params_clean['p'] = rdd_params['p']
    rdd_params_clean['q'] = rdd_params['q']
    rdd_params_clean['coef_rob'] = rdd_params['coef'].values[2][0]
    rdd_params_clean['se_rob'] = rdd_params['se'].values[2][0]
    rdd_params_clean['pv_rob'] = rdd_params['pv'].values[2][0]
    rdd_params_clean['ci_rob_l'] = rdd_params['ci']['CI Lower'].values[2]
    rdd_params_clean['ci_rob_u'] = rdd_params['ci']['CI Upper'].values[2]
    df_rdd = pd.DataFrame()
    for param in rdd_params_clean.keys():
        df_rdd[param] = [rdd_params_clean[param]]
    return df_rdd, rdd_params_clean

def estimate_best_threshold(data_val):
    """
    taken from Mozannar et al. 2023, but applied to all the methods
    :param self:
    :param dataloader:
    :return:
    """
    rej_scores = np.unique(data_val["rej_score"])
    # sort by rejection score
    # get the 100 quantiles for rejection scores
    rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    # for each quantile, get the coverage and accuracy by getting a new deferral decision
    all_metrics = []
    best_treshold = 0
    best_accuracy = 0
    for q in rej_scores_quantiles:
        # get deferral decision
        defers = (data_val["rej_score"] > q).astype(int)
        copy_data = copy.deepcopy(data_val)
        copy_data["defers"] = defers
        # compute metrics
        metrics = compute_deferral_metrics(copy_data)
        if metrics['system_acc'] > best_accuracy:
            best_accuracy = metrics['system_acc']
            best_treshold = q
    return best_treshold

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """_summary_: Updates the average meter with the new value and the number of samples
        Args:
            val (_type_): value
            n (int, optional):  Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """_summary_

    Args:
        output (tensor): output of the model
        target (_type_): target
        topk (tuple, optional): topk. Defaults to (1,).

    Returns:
        float: accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExpertDatasetTensor(Dataset):
    """Generic dataset with expert predictions and labels and images"""

    def __init__(self, images, targets, exp_preds):
        self.images = images
        self.targets = np.array(targets)
        self.exp_preds = np.array(exp_preds)

    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        image = self.images[index]
        expert_pred = self.exp_preds[index]
        return torch.FloatTensor(image), label, expert_pred

    def __len__(self):
        return len(self.targets)