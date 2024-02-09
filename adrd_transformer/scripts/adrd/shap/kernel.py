__all__ = ['KernelExplainer']

import random
import copy
from typing import Any, Type
from sklearn.linear_model import LinearRegression
from scipy.special import binom
from scipy.special import comb
from math import comb, factorial
import torch
import numpy as np
import itertools
Tensor = Type[torch.Tensor]

from . import BaseExplainer

NUM_PERTURBATIONS = 1024 * 2
BATCH_SIZE = 1024

class KernelExplainer(BaseExplainer):

    def __init__(self,
        model: Any,
    ):
        """ ... """
        super().__init__(model)

    def _shap_values_core(self,
        smp: dict[str, Tensor], 
        mask: dict[str, Tensor],
        phi_: dict[str, dict[str, float]],
    ):
        """ ... """
        # get the list of available feature names
        avail = [k for k in mask if mask[k].item() == False]
        M = len(avail)

        # generate binary masks which represent the combinations of all
        # available features. 
        perturbation_masks = self._generate_perturbation_masks(M, NUM_PERTURBATIONS)
        perturbation_masks[-1, :] = True

        # generate perturbation masks suitable for transformer
        perturbation_masks_trf = []
        for i, m in enumerate(perturbation_masks):
            mask_copy = mask.copy()
            for j, v in enumerate(m):
                if v == False:
                    mask_copy[avail[j]] = torch.tensor([True])
            perturbation_masks_trf.append(mask_copy)

        perturbation_masks_trf = {
            k: [
                perturbation_masks_trf[i][k] for i in range(NUM_PERTURBATIONS)
            ] for k in self.model.src_modalities
        }

        perturbation_masks_trf = {
            k: torch.concat(perturbation_masks_trf[k]) for k in self.model.src_modalities
        }

        # repeat input dict
        smps = dict()
        for k, v in smp.items():
            if len(v.shape) == 1:
                smps[k] = smp[k].repeat(NUM_PERTURBATIONS)
            else:
                smps[k] = smp[k].repeat(NUM_PERTURBATIONS, 1)

        # mount inputs to device
        perturbation_masks_trf = {
            k: perturbation_masks_trf[k].to(self.model.device) for k in self.model.src_modalities
        }
        smps = {
            k: smps[k].to(self.model.device) for k in self.model.src_modalities
        }

        # run model
        out_trf = self.model.net_(smps, perturbation_masks_trf)

        # for each label, fit linear regression
        x_linreg = np.array(perturbation_masks, dtype=np.float_)
        w_linreg = [(M - 1) / comb(M, sum(m)) / sum(m) / (M - sum(m)) for m in perturbation_masks]
        w_linreg[-1] = sum(w_linreg[:-1]) * 1e6
        for tgt_k in self.model.tgt_modalities:
            y_linreg = out_trf[tgt_k].detach().cpu().numpy()
            linreg = LinearRegression(fit_intercept=False).fit(x_linreg, y_linreg, w_linreg)
            for i, src_k in enumerate(avail):
                phi_[tgt_k][src_k] = linreg.coef_[i]
            
    def _generate_perturbation_masks(self, n_features, n_samples):
        """ ... """        
        # number of combinations w.r.t. subset sizes
        n_combs = [comb(n_features, i) for i in range(1, n_features)]

        # weights w.r.t. subset sizes
        weights = [(n_features - 1) / (i * (n_features - i)) for i in range(1, n_features)]

        # enumerate subsets
        masks = []
        for subset_size in range(1, (n_features + 1) // 2):
            # check if there are enough samples left to enumerate all subsets of this size
            if n_combs[subset_size - 1] * 2 <= n_samples:
                # generate all combinations of subset_size elements
                for subset in itertools.combinations(range(n_features), subset_size):
                    mask = np.zeros(n_features, dtype=np.bool_)
                    for index in subset:
                        mask[index] = 1
                    masks.append(mask)
                    masks.append(~mask)

                # subtract the number of samples used in this round
                n_samples -= n_combs[subset_size - 1] * 2

                # set the corresponding weights to 0 s.t. they won't be sampled again
                weights[subset_size - 1] = 0
                weights[-subset_size] = 0

        # normalize weights
        weights = np.array(weights) / sum(weights)

        # randomly sample remaining subsets
        while n_samples > 0:
            # choose a subset size based on weights
            subset_size = np.random.choice(np.arange(1, n_features), p=weights)

            # choose a subset of the chosen size
            mask = [0] * n_features
            for index in np.random.choice(range(n_features), size=subset_size, replace=False):
                mask[index] = 1
            masks.append(mask)
            n_samples -= 1

        return np.array(masks, dtype=np.bool_)