from typing import Any, Type, Self
from . import BaseExplainer
import torch
import numpy as np
from torch.utils.data import DataLoader
Tensor = Type[torch.Tensor]

from ..model import Transformer
from ..utils import TransformerTestingDataset
from tqdm import tqdm
import pickle as pkl

class DeepExplainer(BaseExplainer):

    def  __init__(self, 
        model: Transformer,
        data: list[dict[str, Any]] | None = None,
    ) -> None:
        """ ... """
        super().__init__(model)

        # set nn to eval mode
        torch.set_grad_enabled(False)
        self.model.net_.eval()

        if data is None: return

        # initialize dataset and dataloader object
        dat = TransformerTestingDataset(data, self.model.src_modalities)
        ldr = DataLoader(
            dataset = dat,
            batch_size = 1,
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTestingDataset.collate_fn,
        )

        # placeholder
        bkg: dict[str, list[Tensor]] = {k: [] for k in self.model.src_modalities}

        # compute the means of all inputs
        # if a feature is categorical, get the embedding
        # if a feature is numerical, store the exact feature value
        for x_batch, mask in tqdm(ldr):
            # mount data to the proper device
            x_batch = {
                k: x_batch[k].to(self.model.device) 
                for k in self.model.src_modalities
            }

            # get feature embbedings
            out_emb: dict[str, Tensor] = self.model.net_.forward_emb(x_batch)

            for k in self.model.src_modalities:
                # skip if the feature is not present
                if mask[k][0] == True: continue

                # store either exact value or its embedding
                type_ = self.model.src_modalities[k]['type']
                if type_ == 'categorical':
                    bkg[k].append(out_emb[k][0].detach().numpy())
                else:
                    bkg[k].append(x_batch[k][0][0].numpy())

        # print([len(_) for _ in bkg.values()])
        # print([_[:3] for _ in bkg.values()])

        # compute feature means
        self.data_mean = {
            k: sum(bkg[k]) / (len(bkg[k]) + 1e-6)
            for k in self.model.src_modalities
        }

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pkl.dump(self.data_mean, f)

    @classmethod
    def from_ckpt(cls, model, filepath: str) -> Self:
        obj = cls(model, None)
        with open(filepath, 'rb') as f:
            obj.data_mean = pkl.load(f)
        return obj
    
    def shap_values(self, 
        x: list[dict[str, Any]],
    ):
        """ ... """
        # set nn to eval mode, but enable gradient
        torch.set_grad_enabled(True)
        self.model.net_.eval()

        # initialize dataset and dataloader object
        dat = TransformerTestingDataset(x, self.model.src_modalities)
        ldr = DataLoader(
            dataset = dat,
            batch_size = 1,
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTestingDataset.collate_fn,
        )

        # shapley values: phi[sample_idx][tgt][src]
        phi: list[dict[str, dict[str, Tensor]]] = []

        # run model and evaluate shap values
        for x_batch, mask in ldr:
            # mount data to the proper device
            x_batch: dict[str, Tensor] = {
                k: x_batch[k].to(self.model.device) 
                for k in self.model.src_modalities
            }
            mask: dict[str, Tensor] = {
                k: mask[k].to(self.model.device) 
                for k in self.model.src_modalities
            }

            # specify requires_grad_ for numerical features
            for src_k in self.model.src_modalities:
                type_ = self.model.src_modalities[src_k]['type']
                if type_ == 'numerical':
                    x_batch[src_k].requires_grad_()

            # forward
            out_emb: dict[str, Tensor] = self.model.net_.forward_emb(x_batch)
            out_cls: dict[str, Tensor] = self.model.net_.forward_cls(
                self.model.net_.forward_trf(out_emb, mask)
            )

            # specify retain_grad for embeddings
            for src_k in self.model.src_modalities:
                out_emb[src_k].retain_grad()

            # shapley values for one sample: phi_[tgt][src]
            phi_: dict[str, dict[str, Tensor]] = {}

            # backward w.r.t each target/label
            for tgt_k in self.model.tgt_modalities:
                phi_[tgt_k]: dict[str, Tensor] = {}

                # as backward needs to be done for |src_modalities| times for
                # each forward, retain_graph=True must be specified to preserve 
                # itermediate results generated during forward
                self.model.net_.zero_grad()
                out_cls[tgt_k][0].backward(retain_graph=True)

                # loop through sources/features to compute shapley values
                for src_k in self.model.src_modalities:
                    type_ = self.model.src_modalities[src_k]['type']
                    if type_ == 'numerical':
                        grad = x_batch[src_k].grad
                        phi_[tgt_k][src_k] = grad * (
                            x_batch[src_k] - self.data_mean[src_k]
                        )
                        phi_[tgt_k][src_k] = phi_[tgt_k][src_k].item()

                    elif type_ == 'categorical':
                        try:
                            grad = out_emb[src_k].grad
                            phi_[tgt_k][src_k] = torch.inner(
                                grad, x_batch[src_k] - self.data_mean[src_k]
                            )
                            phi_[tgt_k][src_k] = phi_[tgt_k][src_k].item()
                        except:
                            # print(grad.shape, x_batch[src_k].shape, self.data_mean[src_k].shape, src_k)
                            phi_[tgt_k][src_k] = 0.0

                    else:
                        raise ValueError('Unrecognized feature type encountered.')
                    
            phi.append(phi_)
        
        return phi
