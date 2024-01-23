'''
V1
'''

import torch
import torch.nn as nn
from typing import Any, Type
Tensor = Type[torch.Tensor]

from .resnet3d import r3d_18


class Transformer(nn.Module):
    ''' ... '''
    def __init__(self, 
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
    ) -> None:
        ''' ... '''
        super().__init__()

        # embedding modules for source
        self.modules_emb_src = nn.ModuleDict()
        for k, info in src_modalities.items():
            if info['type'] == 'categorical':
                self.modules_emb_src[k] = nn.Embedding(
                    info['num_categories'], d_model
                )
            elif info['type'] == 'numerical':
                self.modules_emb_src[k] = nn.Sequential(
                    nn.BatchNorm1d(info['length']),
                    nn.Linear(info['length'], d_model)
                )
            elif info['type'] == 'imaging' and len(info['shape']) == 4:
                self.modules_emb_src[k] = nn.Sequential(
                    r3d_18(),
                    # nn.Linear(info['length'], d_model)
                    nn.Dropout(0.5)
                )
            else:
                # unrecognized
                raise ValueError('{} is an unrecognized data modality'.format(k))

        # auxiliary embedding vectors for targets
        self.emb_aux = nn.Parameter(
            torch.zeros(len(tgt_modalities), 1, d_model),
            requires_grad = True,
        )

        # transformer
        self.transformer = nn.Transformer(
            d_model, nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = d_model,
            activation = 'gelu',
            dropout = 0.3,
        )

        # classifiers (binary only)
        self.modules_cls = nn.ModuleDict()
        for k, info in tgt_modalities.items():
            if info['type'] == 'categorical' and info['num_categories'] == 2:
                # categorical
                self.modules_cls[k] = nn.Linear(d_model, 1)

            else:
                # unrecognized
                raise ValueError

    def forward(self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        skip_embedding: dict[str, bool] | None = None,
    ) -> dict[str, Tensor]:
        """ ... """
        out_emb = self.forward_emb(x, skip_embedding)
        out_trf = self.forward_trf(out_emb, mask)
        out_cls = self.forward_cls(out_trf)
        return out_cls
    
    def forward_emb(self,
        x: dict[str, Tensor],
        skip_embedding: dict[str, bool] | None = None,
    ) -> dict[str, Tensor]:
        """ ... """
        out_emb = dict()
        for k in self.modules_emb_src.keys():
            if skip_embedding is not None and k in skip_embedding and skip_embedding[k]:
                out_emb[k] = x[k]
            else:
                out_emb[k] = self.modules_emb_src[k](x[k])
        return out_emb
    
    def forward_trf(self,
        out_emb: dict[str, Tensor],
        mask: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        N = len(next(iter(out_emb.values())))  # batch size
        S = len(self.modules_emb_src)  # number of sources
        T = len(self.modules_cls)  # number of targets
        src_iter = self.modules_emb_src.keys()
        tgt_iter = self.modules_cls.keys()

        # stack source embeddings
        emb_src = torch.stack(list(out_emb.values()), dim=0)

        # target embedding
        emb_tgt = self.emb_aux.repeat(1, N, 1)

        # combine masks
        mask = [mask[k] for k in src_iter]
        mask = torch.stack(mask, dim=1)

        # generate src_mask and mem_mask using mask
        src_mask = mask.unsqueeze(1).expand(-1, S, -1).repeat(self.transformer.nhead, 1, 1)
        mem_mask = mask.unsqueeze(1).expand(-1, T, -1).repeat(self.transformer.nhead, 1, 1)
        
        # run transformer
        out_trf = self.transformer(
            emb_src, emb_tgt,
            src_mask = src_mask,
            memory_mask = mem_mask,
        )
        out_trf = {k: out_trf[i] for i, k in enumerate(tgt_iter)}
        return out_trf

    def forward_cls(self,
        out_trf: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        tgt_iter = self.modules_cls.keys()
        out_cls = {k: self.modules_cls[k](out_trf[k]).squeeze(1) for k in tgt_iter}
        return out_cls


if __name__ == '__main__':
    ''' for testing purpose only '''
    pass



