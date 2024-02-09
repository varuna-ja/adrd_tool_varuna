'''
V3: V2 + only 1 letent vector before cls
'''

import torch
import torch.nn as nn
from typing import Any, Type
import math
Tensor = Type[torch.Tensor]

# from .resnet3d import r3d_18


class Transformer(nn.Module):
    ''' ... '''
    def __init__(self, 
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        d_model: int,
        nhead: int,
        num_layers: int,
    ) -> None:
        ''' ... '''
        super().__init__()

        # embedding modules for source
        self.modules_emb_src = nn.ModuleDict()
        for k, info in src_modalities.items():
            if info['type'] == 'categorical':
                self.modules_emb_src[k] = nn.Sequential(
                    nn.Embedding(info['num_categories'], d_model),
                )
            elif info['type'] == 'numerical':
                self.modules_emb_src[k] = nn.Sequential(
                    nn.BatchNorm1d(info['length']),
                    nn.Linear(info['length'], d_model),
                )
            elif info['type'] == 'imaging' and len(info['shape']) == 4:
                self.modules_emb_src[k] = nn.Sequential(
                    # r3d_18(),
                    # nn.Linear(info['length'], d_model)
                    # nn.Dropout(0.5)
                    # TODO
                    nn.Identity()
                )
            else:
                # unrecognized
                raise ValueError('{} is an unrecognized data modality'.format(k))

        # positional encoding
        self.pe = PositionalEncoding(d_model)

        # auxiliary embedding vectors for targets
        num_aux = 1
        self.emb_aux = nn.Parameter(
            torch.zeros(num_aux, 1, d_model),
            requires_grad = True,
        )

        # transformer
        enc = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward = d_model,
            activation = 'gelu',
            dropout = 0.3,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = d_model

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
        T = 1  # number of auxilliary tokens
        src_iter = self.modules_emb_src.keys()

        # stack source embeddings and apply positional encoding
        emb_src = torch.stack(list(out_emb.values()), dim=0)
        emb_src = self.pe(emb_src)

        # target embedding
        emb_tgt = self.emb_aux.repeat(1, N, 1)

        # concatenate source embeddings and target embeddings
        emb_all = torch.concatenate((emb_tgt, emb_src), dim=0)

        # stack source masks
        mask_src = [mask[k] for k in src_iter]
        mask_src = torch.stack(mask_src, dim=1)

        # target masks
        mask_tgt = torch.zeros((N, T), dtype=torch.bool, device=self.emb_aux.device)

        # concatenate source masks and target masks
        mask_all = torch.concatenate((mask_tgt, mask_src), dim=1)

        # repeat mask_all to fit transformer
        mask_all = mask_all.unsqueeze(1).expand(-1, S + T, -1).repeat(self.nhead, 1, 1)
        
        # run transformer
        out_trf = self.transformer(emb_all, mask_all)[0]
        return out_trf

    def forward_cls(self,
        out_trf: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        tgt_iter = self.modules_cls.keys()
        out_cls = {k: self.modules_cls[k](out_trf).squeeze(1) for k in tgt_iter}
        return out_cls
    

class PositionalEncoding(nn.Module):

    def __init__(self, 
        d_model: int, 
        max_len: int = 512
    ):
        """ ... """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return x + self.pe[:x.size(0)]


if __name__ == '__main__':
    ''' for testing purpose only '''
    src_modalities = {
        'A': {
            'type': 'categorical',
            'num_categories': 2,
        },
        'B': {
            'type': 'numerical',
            'length': 1,
        },
        'C': {
            'type': 'numerical',
            'length': 1,
        },
        'D': {
            'type': 'numerical',
            'length': 1,
        },
        'E': {
            'type': 'numerical',
            'length': 1,
        },
    }

    tgt_modalities = {
        'a': {
            'type': 'categorical',
            'num_categories': 2,
        },
        'b': {
            'type': 'categorical',
            'num_categories': 2,
        },
        'c': {
            'type': 'categorical',
            'num_categories': 2,
        },
    }

    trf = Transformer(
        src_modalities, tgt_modalities,
        d_model = 128,
        nhead = 1,
        num_layers = 1,
    )

    # data
    x = {
        'A': torch.randint(0, 2, [7]),
        'B': torch.rand([7, 1]),
        'C': torch.rand([7, 1]),
        'D': torch.rand([7, 1]),
        'E': torch.rand([7, 1])
    }

    mask = {
        'A': torch.randint(0, 2, [7], dtype=torch.bool),
        'B': torch.randint(0, 2, [7], dtype=torch.bool),
        'C': torch.randint(0, 2, [7], dtype=torch.bool),
        'D': torch.randint(0, 2, [7], dtype=torch.bool),
        'E': torch.randint(0, 2, [7], dtype=torch.bool),
    }

    print(trf(x, mask))



