"""
The code is adapted from the following repository:
https://github.com/clinicalml/human_ai_deferral
"""
from abc import ABC, abstractmethod
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import shutil
import time
import torch.utils.data as data
import sys
import pickle
import logging


sys.path.append("..")
from src.utils import *
from src.metrics import *
from tqdm import tqdm

eps_cst = 1e-8

class BaseMethod(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """this function should fit the model and be enough to evaluate the model"""
        pass

    def fit_hyperparam(self, *args, **kwargs):
        """This is an optional method that fits and optimizes hyperparameters over a validation set"""
        return self.fit(*args, **kwargs)

    @abstractmethod
    def test(self, dataloader):
        """this function should return a dict with the following keys:
        'defers': deferred binary predictions
        'preds':  classifier predictions
        'labels': labels
        'hum_preds': human predictions
        'rej_score': a real score for the rejector, the higher the more likely to be rejected
        'class_probs': probability of the classifier for each class (can be scores as well)
        """
        pass


class BaseSurrogateMethod(BaseMethod):
    """Abstract method for learning to defer methods based on a surrogate model"""

    def __init__(self, alpha, plotting_interval, model, device, learnable_threshold_rej=False):
        '''
        alpha: hyperparameter for surrogate loss
        plotting_interval (int): used for plotting model training in fit_epoch
        model (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the reject score (applicable to RealizableSurrogate only)
        '''
        self.alpha = alpha
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device
        self.threshold_rej = 0
        self.learnable_threshold_rej = learnable_threshold_rej

    @abstractmethod
    def surrogate_loss_function(self, outputs, hum_preds, data_y, weight=None):
        """surrogate loss function"""
        pass

    def fit_epoch(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """
        Fit the model for one epoch
        model: model to be trained
        dataloader: dataloader
        optimizer: optimizer
        verbose: print loss
        epoch: epoch number
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs = self.model(data_x)
            loss = self.surrogate_loss_function(outputs, hum_preds, data_y, weight=weight)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def fit(
            self,
            dataloader_train,
            dataloader_val,
            dataloader_test,
            epochs,
            optimizer,
            lr,
            scheduler=None,
            verbose=True,
            test_interval=1,
            step_size=25,
            gamma=0.1,
            m_norm=5,
            weight=None
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            # scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
            scheduler = scheduler(optimizer, step_size, gamma=gamma)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch, m_norm=m_norm, weight=weight)
            if epoch % test_interval == 0 and epoch > 1:
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())
                if verbose:
                    logging.info(compute_deferral_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
                if epoch % 10 == 0:
                    print("{}".format(optimizer.param_groups[0]['lr']))
        self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test)
        return compute_deferral_metrics(final_test)

    def fit_treshold_rej(self, dataloader):
        data_test = self.test(dataloader)
        rej_scores = np.unique(data_test["rej_score"])
        # sort by rejection score
        # get the 100 quantiles for rejection scores
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
        # for each quantile, get the coverage and accuracy by getting a new deferral decision
        all_metrics = []
        best_treshold = 0
        best_accuracy = 0
        for q in rej_scores_quantiles:
            # get deferral decision
            defers = (data_test["rej_score"] > q).astype(int)
            copy_data = copy.deepcopy(data_test)
            copy_data["defers"] = defers
            # compute metrics
            metrics = compute_deferral_metrics(copy_data)
            if metrics['system_acc'] > best_accuracy:
                best_accuracy = metrics['system_acc']
                best_treshold = q
        self.threshold_rej = best_treshold

    def test(self, dataloader):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs = self.model(data_x)
                outputs_class = F.softmax(outputs[:, :-1], dim=1)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)
                predictions_all.extend(predicted_class.cpu().numpy())

                defer_scores = [outputs.data[i][-1].item() - outputs.data[i][predicted_class[i]].item() for i in
                                range(len(outputs.data))]
                defer_binary = [int(defer_score >= self.threshold_rej) for defer_score in defer_scores]
                defers_all.extend(defer_binary)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                for i in range(len(outputs.data)):
                    rej_score_all.append(
                        outputs.data[i][-1].item()
                        - outputs.data[i][predicted_class[i]].item()
                    )
                class_probs_all.extend(outputs_class.cpu().numpy())

        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data

class CompareConfidence(BaseMethod):
    """Method trains classifier indepedently on cross entropy,
    and expert model on whether human prediction is equal to ground truth.
    Then, at each test point we compare the confidence of the classifier
    and the expert model.
    """

    def __init__(self, model_class, model_expert, device, plotting_interval=100):
        """
        Args:
            model_class (pytorch model): _description_
            model_expert (pytorch model): _description_
            device (str): device
            plotting_interval (int, optional): _description_. Defaults to 100.
        """
        self.model_class = model_class
        self.model_expert = model_expert
        self.device = device
        self.plotting_interval = plotting_interval

    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """
        train classifier for single epoch
        Args:
            dataloader (dataloader): _description_
            optimizer (optimizer): _description_
            verbose (bool, optional): to print loss or not. Defaults to True.
            epoch (int, optional): _description_. Defaults to 1.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss()

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            outputs = self.model_class(data_x)
            # cross entropy loss
            loss = F.cross_entropy(outputs, data_y, weight=weight)
            torch.nn.utils.clip_grad_norm_(self.model_class.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def fit_epoch_expert(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """train expert model for single epoch

        Args:
            dataloader (_type_): _description_
            optimizer (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.
            epoch (int, optional): _description_. Defaults to 1.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.model_expert.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            hum_equal_to_y = (hum_preds == data_y).long()
            hum_equal_to_y = (hum_equal_to_y).clone().to(self.device)
            outputs = self.model_expert(data_x)
            # cross entropy loss
            loss = loss_fn(outputs, hum_equal_to_y)
            torch.nn.utils.clip_grad_norm_(self.model_expert.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, hum_equal_to_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
        step_size=25,
        gamma=0.1,
        m_norm=5,
        weight=None
    ):
        """fits classifier and expert model

        Args:
            dataloader_train (_type_): train dataloader
            dataloader_val (_type_): val dataloader
            dataloader_test (_type_): _description_
            epochs (_type_): training epochs
            optimizer (_type_): optimizer function
            lr (_type_): learning rate
            scheduler (_type_, optional): scheduler function. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
            test_interval (int, optional): _description_. Defaults to 5.

        Returns:
            dict: metrics on the test set
        """
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        optimizer_expert = optimizer(self.model_expert.parameters(), lr=lr)
        if scheduler is not None:
            # scheduler_class = scheduler(optimizer_class, len(dataloader_train) * epochs)
            scheduler_class = scheduler(optimizer_class, step_size, gamma=gamma)
            # scheduler_expert = scheduler(
            #     optimizer_expert, len(dataloader_train) * epochs
            # )
            scheduler_expert = scheduler(optimizer_expert, step_size, gamma=gamma)
        best_acc = 0
        # store current model dict
        best_model = [copy.deepcopy(self.model_class.state_dict()), copy.deepcopy(self.model_expert.state_dict())]
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch, m_norm=m_norm, weight=weight
            )
            self.fit_epoch_expert(
                dataloader_train, optimizer_expert, verbose=verbose, epoch=epoch, m_norm=m_norm, weight=weight
            )
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["classifier_all_acc"] >= best_acc:
                    best_acc = val_metrics["classifier_all_acc"]
                    best_model = [copy.deepcopy(self.model_class.state_dict()), copy.deepcopy(self.model_expert.state_dict())]

            if scheduler is not None:
                scheduler_class.step()
                scheduler_expert.step()
                if epoch % 10 == 0:
                    print("{}".format(optimizer_class.param_groups[0]["lr"]))
        self.model_class.load_state_dict(best_model[0])
        self.model_expert.load_state_dict(best_model[1])

        return compute_deferral_metrics(self.test(dataloader_test))

    def test(self, dataloader):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_expert.eval()
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                outputs_expert = self.model_expert(data_x)
                outputs_expert = F.softmax(outputs_expert, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                class_probs_all.extend(outputs_class.cpu().numpy())
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                defers = []
                for i in range(len(data_y)):
                    rej_score_all.extend(
                        [outputs_expert[i, 1].item() - max_class_probs[i].item()]
                    )
                    if outputs_expert[i, 1] > max_class_probs[i]:
                        defers.extend([1])
                    else:
                        defers.extend([0])
                defers_all.extend(defers)
        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data

class DifferentiableTriage(BaseMethod):
    def __init__(
        self,
        model_class,
        model_rejector,
        device,
        weight_low=0.00,
        strategy="human_error",
        plotting_interval=100,
    ):
        """Method from the paper 'Differentiable Learning Under Triage' adapted to this setting
        Args:
            model_class (_type_): _description_
            model_rejector (_type_): _description_
            device (_type_): _description_
            weight_low (float in [0,1], optional): weight for points that are deferred so that classifier trains less on them
            strategy (_type_): pick between "model_first", "human_error"
                "model_first" means that the rejector is 1 only if the human is correct and the model is wrong
                "human_error": the rejector is 1 if the human gets it right, otherwise 0
            plotting_interval (int, optional): _description_. Defaults to 100.

        """
        self.model_class = model_class
        self.model_rejector = model_rejector
        self.device = device
        self.weight_low = weight_low
        self.plotting_interval = plotting_interval
        self.strategy = strategy

    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """
        train classifier for single epoch
        Args:
            dataloader (dataloader): _description_
            optimizer (optimizer): _description_
            verbose (bool, optional): to print loss or not. Defaults to True.
            epoch (int, optional): _description_. Defaults to 1.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            outputs = self.model_class(data_x)
            # cross entropy loss
            loss = F.cross_entropy(outputs, data_y, weight=weight)
            torch.nn.utils.clip_grad_norm_(self.model_class.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def find_machine_samples(self, model_outputs, data_y, hum_preds):
        """

        Args:
            model_outputs (_type_): _description_
            data_y (_type_): _description_
            hum_preds (_type_): _description_

        Returns:
            array:  binary array of size equal to the input indicating whether to train or not on each poin
        """
        max_class_probs, predicted_class = torch.max(model_outputs.data, 1)
        model_error = predicted_class != data_y
        hum_error = hum_preds != data_y
        rejector_labels = []
        soft_weights_classifier = []
        if self.strategy == "model_first":
            for i in range(len(model_outputs)):
                if not model_error[i]:
                    rejector_labels.append(0)
                    soft_weights_classifier.append(1)
                elif not hum_error[i]:
                    rejector_labels.append(1)
                    soft_weights_classifier.append(self.weight_low)
                else:
                    rejector_labels.append(0)
                    soft_weights_classifier.append(1.0)
        else:
            for i in range(len(model_outputs)):
                if not hum_error[i]:
                    rejector_labels.append(1)
                    soft_weights_classifier.append(self.weight_low)
                else:
                    rejector_labels.append(0)
                    soft_weights_classifier.append(1.0)

        rejector_labels = torch.tensor(rejector_labels).long().to(self.device)
        soft_weights_classifier = torch.tensor(soft_weights_classifier).to(self.device)
        return rejector_labels, soft_weights_classifier

    def fit_epoch_class_triage(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """
        Fit the model for classifier for one epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs = self.model_class(data_x)
            # cross entropy loss

            rejector_labels, soft_weights_classifier = self.find_machine_samples(
                outputs, data_y, hum_preds
            )

            loss = weighted_cross_entropy_loss(outputs, data_y, soft_weights_classifier, weight_class=weight)
            torch.nn.utils.clip_grad_norm_(self.model_class.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def fit_epoch_rejector(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """
        Fit the rejector for one epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.model_rejector.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs_class = self.model_class(data_x)
            rejector_labels, soft_weights_classifier = self.find_machine_samples(
                outputs_class, data_y, hum_preds
            )
            outputs = self.model_rejector(data_x)
            # cross entropy loss
            loss = F.cross_entropy(outputs, rejector_labels, weight=weight)
            torch.nn.utils.clip_grad_norm_(self.model_rejector.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, rejector_labels, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler=None,
        step_size=25,
        gamma=0.1,
        m_norm=5,
        weight=None
    ):
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        optimizer_rejector = optimizer(self.model_rejector.parameters(), lr=lr)
        if scheduler is not None:
            # scheduler_class = scheduler(optimizer_class, len(dataloader_train) * epochs)
            scheduler_class = scheduler(optimizer_class, step_size, gamma=gamma)
            # scheduler_rejector = scheduler(optimizer_rejector, len(dataloader_train) * epochs)
            scheduler_rejector = scheduler(optimizer_rejector, step_size, gamma=gamma)
        self.model_class.train()
        self.model_rejector.train()

        logging.info("Re-training classifier on data based on the formula")
        for epoch in tqdm(range(int(epochs))):
            self.fit_epoch_class_triage(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch, m_norm=m_norm, weight=weight
            )
            if verbose and epoch % test_interval == 0:
                logging.info(compute_classification_metrics(self.test(dataloader_val)))
            if scheduler is not None:
                scheduler_class.step()
        # now fit rejector

        logging.info("Fitting rejector on all data")
        best_acc = 0
        best_model = copy.deepcopy(self.model_rejector.state_dict())

        for epoch in tqdm(range(int(epochs))):
            self.fit_epoch_rejector(
                dataloader_train, optimizer_rejector, verbose=verbose, epoch=epoch, m_norm=m_norm, weight=weight
            )
            if verbose and epoch % test_interval == 0:
                logging.info(compute_deferral_metrics(self.test(dataloader_val)))
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model_rejector.state_dict())

            if scheduler is not None:
                scheduler_rejector.step()
        self.model_rejector.load_state_dict(best_model)
        return compute_deferral_metrics(self.test(dataloader_test))

    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler  = None,
        step_size = 25,
        gamma = 0.1,
        path_model_save = "differentiable_triage_",
        m_norm = 5,
        weight = None
    ):
        weight_low_grid = [0,  1]
        best_weight = 0
        best_acc = 0
        model_rejector_dict = copy.deepcopy(self.model_rejector.state_dict())
        model_class_dict = copy.deepcopy(self.model_class.state_dict())
        for weight_ in tqdm(weight_low_grid):
            self.weight_low = weight_
            self.model_rejector.load_state_dict(model_rejector_dict)
            self.model_class.load_state_dict(model_class_dict)
            self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
                step_size=step_size,
                gamma=gamma,
                m_norm=m_norm,
                weight=weight
            )["system_acc"]
            accuracy = compute_deferral_metrics(self.test(dataloader_val))["system_acc"]
            logging.info(f"weight low : {weight_}, accuracy: {accuracy}")
            torch.save(self.model_rejector.state_dict(), path_model_save + f"_rej_weight_{weight_}.pt")
            torch.save(self.model_class.state_dict(), path_model_save + f"_class_weight_{weight_}.pt")
            if accuracy > best_acc:
                best_acc = accuracy
                best_weight = weight_
        self.weight_low = best_weight
        self.model_rejector.load_state_dict(model_rejector_dict)
        self.model_class.load_state_dict(model_class_dict)
        fit = self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
            step_size=step_size,
            gamma=gamma,
            m_norm=m_norm,
            weight=weight
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

    def test(self, dataloader):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_rejector.eval()
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                outputs_rejector = self.model_rejector(data_x)
                outputs_rejector = F.softmax(outputs_rejector, dim=1)
                _, predictions_rejector = torch.max(outputs_rejector.data, 1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                defers_all.extend(predictions_rejector.cpu().numpy())
                rej_score_all.extend(outputs_rejector[:, 1].cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data

class LceSurrogate(BaseSurrogateMethod):
    def surrogate_loss_function(self, outputs, hum_preds, data_y, weight=None):
        """
        Implmentation of L_{CE}^{\alpha}
        """
        outputs = F.softmax(outputs, dim=1)
        human_correct = (hum_preds == data_y).float()
        m2 = self.alpha * human_correct + (1 - human_correct)
        human_correct = torch.tensor(human_correct).to(self.device)
        m2 = torch.tensor(m2).to(self.device)
        batch_size = outputs.size()[0]  # batch_size
        loss = -human_correct * torch.log2(
            outputs[range(batch_size), -1] + eps_cst
        ) - m2 * torch.log2(
            outputs[range(batch_size), data_y] + eps_cst
        )
        #il peso va fuori dal logaritmo
        if weight is not None:
            weights = torch.tensor(weight)
            w_vect = weights.expand((batch_size, outputs.shape[1] - 1))
            if len(data_y.shape) == 1:
                w_single = torch.gather(w_vect, 1, data_y.unsqueeze(1)).float().to(self.device)
            else:
                w_single = torch.gather(w_vect, 1, data_y).float().to(self.device)
            if len(loss.shape) != len(w_single.shape):
                loss = w_single.reshape(-1) * loss
            else:
                loss = loss * w_single
        return torch.sum(loss) / batch_size

    # fit with hyperparameter tuning over alpha
    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
            step_size=25,
            gamma=0.1,
            path_model_save = "lce_surrogate_",
            m_norm = 5,
            weight = None
    ):
        alpha_grid = [0, 0.5, 1]
        best_alpha = 0
        best_acc = 0
        model_dict = copy.deepcopy(self.model.state_dict())
        for alpha in tqdm(alpha_grid):
            self.alpha = alpha
            self.model.load_state_dict(model_dict)
            self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
                step_size=step_size,
                gamma=gamma,
                m_norm=m_norm,
                weight=weight
            )["system_acc"]
            accuracy = compute_deferral_metrics(self.test(dataloader_val))["system_acc"]
            logging.info(f"alpha: {alpha}, accuracy: {accuracy}")
            torch.save(self.model.state_dict(), path_model_save + f"alpha_{alpha}.pt")
            if accuracy > best_acc:
                best_acc = accuracy
                best_alpha = alpha
        self.alpha = best_alpha
        self.model.load_state_dict(model_dict)
        fit = self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
            step_size=step_size,
            gamma=gamma,
            m_norm=m_norm,
            weight=weight
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

class MixtureOfExperts(BaseMethod):
    """Implementation of Madras et al., 2018"""

    def __init__(self, model, device, plotting_interval=100):
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device

    def mixtures_of_experts_loss(self, outputs, human_is_correct, labels, weight=None):
        """
        Implmentation of Mixtures of Experts loss from Madras et al., 2018
        """

        batch_size = outputs.size()[0]  # batch_size
        human_loss = torch.Tensor((1 - human_is_correct * 1.0)).clone().float().to(self.device)
        rejector_probability = torch.sigmoid(
            outputs[:, -1] + eps_cst
        )  # probability of rejection
        outputs_class = F.softmax(outputs[:, :-1], dim=1)
        classifier_loss = -torch.log2(
            outputs_class[range(batch_size), labels] + eps_cst
        )
        loss = (
            classifier_loss * (1 - rejector_probability)
            + human_loss * rejector_probability
        )
        if weight is not None:
            weights = torch.tensor(weight)
            w_vect = weights.expand((batch_size, outputs.shape[1] - 1))
            if len(labels.shape) == 1:
                w_single = torch.gather(w_vect, 1, labels.unsqueeze(1)).float().to(self.device)
            else:
                w_single = torch.gather(w_vect, 1, labels).float().to(self.device)
            if len(loss.shape) != len(w_single.shape):
                loss = w_single.reshape(-1) * loss
            else:
                loss = loss * w_single
        return torch.sum(loss) / batch_size

    def fit_epoch(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        """
        Fit the model for one epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            m = (hum_preds == data_y) * 1
            m = torch.tensor(m).to(self.device)
            outputs = self.model(data_x)
            # apply softmax to outputs
            loss = self.mixtures_of_experts_loss(outputs, m, data_y, weight=weight)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler=None,
        step_size=25,
        gamma=0.1,
        m_norm=5,
        weight=None
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            # scheduler = scheduler(optimizer, len(dataloader_train)*epochs)
            scheduler = scheduler(optimizer, step_size, gamma=gamma)
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch, m_norm=m_norm, weight=weight)
            if verbose and epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val)
                logging.info(compute_deferral_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
                if epoch % 10 == 0:
                    print("{}".format(optimizer.param_groups[0]['lr']))
        final_test = self.test(dataloader_test)
        return compute_deferral_metrics(final_test)

    def test(self, dataloader):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        rej_score = []
        predictions_all = []  # classifier only
        class_probs_all = []
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs = self.model(data_x)
                outputs_soft = F.softmax(outputs[:, :-1], dim=1)
                class_probs_all.extend(outputs_soft.cpu().numpy())
                _, predicted_class = torch.max(outputs_soft.data, 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                rejector_outputs = torch.sigmoid(outputs[:, -1])
                defers_all.extend((rejector_outputs.cpu().numpy() >= 0.5).astype(int))
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                rej_score.extend(rejector_outputs.cpu().numpy())
        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score,
            "class_probs": class_probs_all
        }
        return data

class OVASurrogate(BaseSurrogateMethod):
    """Method of OvA surrogate from Calibrated Learning to Defer with One-vs-All Classifiers https://proceedings.mlr.press/v162/verma22c/verma22c.pdf"""

    # from https://github.com/rajevv/OvA-L2D/blob/main/losses/losses.py
    def LogisticLossOVA(self, outputs, y):
        outputs[torch.where(outputs == 0.0)] = (-1 * y) * (-1 * np.inf)
        l = torch.log2(1 + torch.exp((-1 * y) * outputs + eps_cst) + eps_cst)
        return l

    def surrogate_loss_function(self, outputs, hum_preds, data_y, weight=None):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting  hum_preds == target
        labels: target
        """
        human_correct = (hum_preds == data_y).float()
        human_correct = torch.tensor(human_correct).to(self.device)
        batch_size = outputs.size()[0]
        l1 = self.LogisticLossOVA(outputs[range(batch_size), data_y], 1)
        l2 = torch.sum(
            self.LogisticLossOVA(outputs[:, :-1], -1), dim=1
        ) - self.LogisticLossOVA(outputs[range(batch_size), data_y], -1)
        l3 = self.LogisticLossOVA(outputs[range(batch_size), -1], -1)
        l4 = self.LogisticLossOVA(outputs[range(batch_size), -1], 1)

        l5 = human_correct * (l4 - l3)

        l = l1 + l2 + l3 + l5
        if weight is not None:
            weights = torch.tensor(weight)
            w_vect = weights.expand((batch_size, outputs.shape[1] - 1))
            if len(data_y.shape) == 1:
                w_single = torch.gather(w_vect, 1, data_y.unsqueeze(1)).float().to(self.device)
            else:
                w_single = torch.gather(w_vect, 1, data_y).float().to(self.device)
            if len(l.shape) != len(w_single.shape):
                l = w_single.reshape(-1) * l
            else:
                l = l * w_single
        return torch.mean(l)

class SelectivePrediction(BaseMethod):
    """Selective Prediction method, train classifier on all data, and defer based on thresholding classifier confidence (max class prob)"""

    def __init__(self, model_class, device, plotting_interval=1):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        self.treshold_class = 0.5

    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1, m_norm=5, weight=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        self.model_class.train()
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            outputs = self.model_class(data_x)
            # cross entropy loss
            loss = F.cross_entropy(outputs, data_y, weight=weight)
            torch.nn.utils.clip_grad_norm_(self.model_class.parameters(), max_norm=m_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break

    def set_optimal_threshold(self, dataloader):
        """set threshold to maximize system accuracy on validation set

        Args:
            dataloader (_type_): dataloader validation set
        """
        data_preds = self.test(dataloader)
        treshold_grid = data_preds["max_probs"]
        treshold_grid = np.append(treshold_grid, np.linspace(0, 1, 20))
        best_treshold = 0
        best_acc = 0
        # optimize for system accuracy
        for treshold in treshold_grid:
            defers = (data_preds["max_probs"] < treshold) * 1
            acc = sklearn.metrics.accuracy_score(
                data_preds["preds"] * (1 - defers) + data_preds["hum_preds"] * (defers),
                data_preds["labels"],
            )
            if acc > best_acc:
                best_acc = acc
                best_treshold = treshold
        logging.info(f"Best treshold {best_treshold} with accuracy {best_acc}")
        self.treshold_class = best_treshold

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=1,
        scheduler=None,
        step_size=25,
        gamma=0.1,
        m_norm=5,
        weight=None
    ):
        # fit classifier and expert same time
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer_class, step_size, gamma=gamma)
            # scheduler = scheduler(optimizer, len(dataloader_train)*epochs)

        self.model_class.train()
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch, m_norm=m_norm, weight=weight
            )
            if verbose and epoch % test_interval == 0:
                logging.info(compute_classification_metrics(self.test(dataloader_val)))
            if scheduler is not None:
                scheduler.step()
        self.set_optimal_threshold(dataloader_val)

        return compute_deferral_metrics(self.test(dataloader_test))

    def test(self, dataloader):
        defers_all = []
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                defers = []
                max_probs.extend(max_class_probs.cpu().numpy())
                for i in range(len(data_y)):
                    rej_score_all.extend([1 - max_class_probs[i].item()])
                    if max_class_probs[i] < self.treshold_class:
                        defers.extend([1])
                    else:
                        defers.extend([0])
                defers_all.extend(defers)
        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "max_probs": max_probs,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data

class RealizableSurrogate(BaseSurrogateMethod):
    def surrogate_loss_function(self, outputs, hum_preds, data_y, weight: torch.tensor or list = None):
        """ Implementation of our RealizableSurrogate loss function
        we added a weight parameter to deal with chestxray dataset, which is highly imbalanced
        """
        human_correct = (hum_preds == data_y).float()
        human_correct = torch.tensor(human_correct).to(self.device)
        batch_size = outputs.size()[0]  # batch_size
        outputs_exp = torch.exp(outputs)
        new_loss = -torch.log2(
            (
                human_correct * outputs_exp[range(batch_size), -1]
                + outputs_exp[range(batch_size), data_y]
            )
            / (torch.sum(outputs_exp, dim=1) + eps_cst)
        )  # pick the values corresponding to the labels
        ce_loss = -torch.log2(
            (outputs_exp[range(batch_size), data_y])
            / (torch.sum(outputs_exp[range(batch_size), :-1], dim=1) + eps_cst)
        )
        loss = self.alpha * new_loss + (1 - self.alpha) * ce_loss
        if weight is not None:
            weights = torch.tensor(weight)
            w_vect = weights.expand((batch_size, outputs.shape[1] - 1))
            if len(data_y.shape) == 1:
                w_single = torch.gather(w_vect, 1, data_y.unsqueeze(1)).float().to(self.device)
            else:
                w_single = torch.gather(w_vect, 1, data_y).float().to(self.device)
            if len(loss.shape) != len(w_single.shape):
                loss = w_single.reshape(-1) * loss
            else:
                loss = w_single * loss
        return torch.sum(loss) / batch_size

    # fit with hyperparameter tuning over alpha
    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=1,
        scheduler=None,
        step_size=25,
        gamma=0.1,
        alpha_grid=[0, 0.1, 0.3, 0.5, 0.9, 1],
        path_model_save="realizable_surrogate_",
        m_norm = 5,
        weight=None
    ):
        # np.linspace(0,1,11)
        best_alpha = 0
        best_acc = 0
        model_dict = copy.deepcopy(self.model.state_dict())
        for alpha in tqdm(alpha_grid):
            self.alpha = alpha
            self.model.load_state_dict(model_dict)
            self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs = epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
                step_size=step_size,
                gamma=gamma,
                m_norm=m_norm,
                weight=weight
            )["system_acc"]
            accuracy = compute_deferral_metrics(self.test(dataloader_val))["system_acc"]
            print(accuracy)
            logging.info(f"alpha: {alpha}, accuracy: {accuracy}")
            torch.save(self.model.state_dict(), path_model_save + f"alpha_{alpha}.pt")
            if (accuracy > best_acc) & (accuracy != np.nan) & (accuracy != np.inf) & (accuracy != -np.inf):
                best_acc = accuracy
                best_alpha = alpha
        self.alpha = best_alpha
        self.model.load_state_dict(model_dict)
        fit = self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs = epochs,
                optimizer = optimizer,
                lr = lr,
                verbose = verbose,
                test_interval = test_interval,
                scheduler = scheduler,
                step_size=step_size,
                gamma=gamma,
                m_norm=m_norm,
                weight=weight
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

class AsymmetricLCESurrogate(BaseSurrogateMethod):
        def Asym_SM_trans(self, outputs):
            class_num = outputs.size()[1]-1
            classifier_input = outputs[:, 0:class_num]
            classifier_input = classifier_input.to(self.device)
            output1 = torch.softmax(classifier_input, dim=-1).to(self.device)
            sm = torch.softmax(outputs, dim=-1).to(self.device)
            rejector_output = sm[:, class_num].view(-1,1)
            norm = -(torch.max(sm[:, 0:class_num],dim=-1)[0].view(-1,1)-1)
            output2 = rejector_output/(norm+1e-7)
            return torch.cat((output1, output2), dim=-1)
        def surrogate_loss_function(self, outputs, hum_preds, data_y, weight=None):
            """
            Implmentation of L_{CE}^{\alpha}
            """
            """
            output = output.cuda()
            label = label.cuda()
            expert_pred = expert.predict(labels=label, input=[])
            expert_pred = torch.tensor(expert_pred)
            expert_pred = expert_pred.cuda()
            expert_correctness = expert_pred == label
            expert_correctness = torch.tensor(expert_correctness)
        
            output_probit = Asym_SM_trans(output)
            output_probit = output_probit.cuda()
            num_class = 100
        
        
            sm = output_probit[:, 0:num_class]
            bsm = output_probit[:, num_class]
            loss1 = torch.log(sm+1e-7)
            loss1 = -loss1.gather(-1, label.view(-1, 1))
            loss2 = -torch.mul(expert_correctness.float(), torch.log(bsm+1e-7))-torch.mul(1-expert_correctness.float(), torch.log(1-bsm+1e-7))
            return torch.mean(loss1+loss2)
            """

            outputs = F.softmax(outputs, dim=1)
            human_correct = (hum_preds == data_y).float()
            outputs_probit = self.Asym_SM_trans(outputs).to(self.device)
            num_class = outputs_probit.size()[1] - 1
            sm = outputs_probit[:, 0:num_class]
            bsm = outputs_probit[:, num_class]
            loss1 = torch.log(sm + eps_cst)
            loss1 = -loss1.gather(-1, data_y.view(-1, 1))
            loss2 = -torch.mul(
                human_correct, torch.log(bsm + eps_cst)
            ) - torch.mul(
                (1 - human_correct), torch.log(1 - bsm + eps_cst)
            )
            loss = loss1+loss2
            batch_size = outputs.size()[0]
            # il peso va fuori dal logaritmo
            if weight is not None:
                weights = torch.tensor(weight)
                w_vect = weights.expand((batch_size, outputs.shape[1] - 1))
                if len(data_y.shape) == 1:
                    w_single = torch.gather(w_vect, 1, data_y.unsqueeze(1)).float().to(self.device)
                else:
                    w_single = torch.gather(w_vect, 1, data_y).float().to(self.device)
                if len(loss.shape) != len(w_single.shape):
                    loss = w_single.reshape(-1) * loss
                else:
                    loss = loss * w_single
            return torch.mean(loss)

