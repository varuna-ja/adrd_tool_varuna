import torch
from torch.utils.data import Dataset
import numpy as np
from functools import cached_property
from typing import Any, Type
from numpy.typing import NDArray
import random

Tensor = Type[torch.Tensor]

from .masker import Masker
from . import DropoutMasker
from . import MissingMasker

from .imputer import Imputer
from . import FrequencyImputer
from . import ConstantImputer
from . import Formatter


class TransformerDataset(Dataset):

    def __init__(self,
        src: list[dict[str, Any]], 
        tgt: list[dict[str, Any]] | None,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]] | None,
        is_embedding: dict[str, bool] | None = None
    ) -> None:
        ''' ... '''
        # boolean dict to indicate which features are embeddings
        self.is_embedding = is_embedding

        # format source
        formatter_src = Formatter(src_modalities)
        self.src = [formatter_src(smp) for smp in src]
        self.src_modalities = src_modalities

        # format target
        if tgt is None: return
        formatter_tgt = Formatter(tgt_modalities)
        self.tgt = [formatter_tgt(smp) for smp in tgt]
        self.tgt_modalities = tgt_modalities

    def __len__(self) -> int:
        ''' ... '''
        return len(self.src)
    
    def __getitem__(self,
        idx: int
    ) -> tuple[
        dict[str, int | NDArray[np.float32]],
        dict[str, int | NDArray[np.float32]],
        dict[str, bool],
        dict[str, bool],
    ]:
        ''' ... '''
        # impute x and y
        x_imp = self.imputer_src(self.src[idx])
        mask_x = self.masker_src(self.src[idx])
        y_imp = self.imputer_tgt(self.tgt[idx]) if hasattr(self, 'tgt') else None
        mask_y = self.masker_tgt(self.tgt[idx]) if hasattr(self, 'tgt') else None

        # replace mmap object by the loaded one
        for k, v in x_imp.items():
            if isinstance(v, np.memmap):
                x_imp[k] = np.load(v.filename)
                x_imp[k] = np.reshape(x_imp[k], v.shape)

        return x_imp, y_imp, mask_x, mask_y
    
    @cached_property
    def imputer_src(self) -> Imputer:
        ''' imputer object '''
        raise NotImplementedError
    
    @cached_property
    def imputer_tgt(self) -> Imputer:
        ''' imputer object '''
        raise NotImplementedError

    @cached_property
    def masker_src(self) -> Masker:
        ''' mask generator object '''
        raise NotImplementedError

    @cached_property
    def masker_tgt(self) -> Masker:
        ''' mask generator object '''
        raise NotImplementedError
    
    @staticmethod
    def collate_fn(
        batch: list[
            tuple[
                dict[str, int | NDArray[np.float32]],
                dict[str, int | NDArray[np.float32]],
                dict[str, bool],
                dict[str, bool],
            ]
        ]
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        ''' ... '''
        # seperate entries
        x = [smp[0] for smp in batch]
        y = [smp[1] for smp in batch]
        m = [smp[2] for smp in batch]
        n = [smp[3] for smp in batch]

        # stack and convert to tensor
        x = {k: torch.as_tensor(np.array([x[i][k] for i in range(len(x))])) for k in x[0]}
        y = {k: torch.as_tensor(np.array([y[i][k] for i in range(len(y))])) for k in y[0]}
        m = {k: torch.as_tensor(np.array([m[i][k] for i in range(len(m))])) for k in m[0]}
        n = {k: torch.as_tensor(np.array([n[i][k] for i in range(len(n))])) for k in n[0]}

        return x, y, m, n


class TransformerTrainingDataset(TransformerDataset):

    def __init__(self,
        src: list[dict[str, Any]], 
        tgt: list[dict[str, Any]],
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        dropout_rate: float = .5,
        dropout_strategy: str = 'compensated',
    ) -> None:
        ''' ... '''
        # call the constructor of parent class
        super().__init__(src, tgt, src_modalities, tgt_modalities)

        # other object parameters
        self.dropout_rate = dropout_rate
        self.dropout_strategy = dropout_strategy

    @cached_property
    def imputer_src(self) -> FrequencyImputer:
        ''' imputer object '''
        return FrequencyImputer(self.src_modalities, self.src)
    
    @cached_property
    def imputer_tgt(self) -> ConstantImputer:
        ''' imputer object '''
        return ConstantImputer(self.tgt_modalities)

    @cached_property
    def masker_src(self) -> DropoutMasker:
        ''' mask generator object '''
        return DropoutMasker(
            self.src_modalities, self.src,
            dropout_rate = self.dropout_rate,
            dropout_strategy = self.dropout_strategy,
        )
    
    @cached_property
    def masker_tgt(self) -> MissingMasker:
        ''' mask generator object '''
        return MissingMasker(self.tgt_modalities)
        

class TransformerValidationDataset(TransformerDataset):

    def __init__(self,
        src: list[dict[str, Any]], 
        tgt: list[dict[str, Any]],
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        is_embedding: dict[str, bool] | None = None
    ) -> None:
        ''' ... '''
        # call the constructor of parent class
        super().__init__(src, tgt, src_modalities, tgt_modalities, is_embedding)

    @cached_property
    def imputer_src(self) -> ConstantImputer:
        ''' imputer object '''
        return ConstantImputer(self.src_modalities, self.is_embedding)
    
    @cached_property
    def imputer_tgt(self) -> ConstantImputer:
        ''' imputer object '''
        return ConstantImputer(self.tgt_modalities)

    @cached_property
    def masker_src(self) -> MissingMasker:
        ''' mask generator object '''
        return MissingMasker(self.src_modalities)
    
    @cached_property
    def masker_tgt(self) -> MissingMasker:
        ''' mask generator object '''
        return MissingMasker(self.tgt_modalities)
    

class TransformerTestingDataset(TransformerValidationDataset):

    def __init__(self,
        src: list[dict[str, Any]], 
        src_modalities: dict[str, dict[str, Any]],
        is_embedding: dict[str, bool] | None = None
    ) -> None:
        ''' ... '''
        # call the constructor of parent class
        super().__init__(src, None, src_modalities, None, is_embedding)

    def __getitem__(self,
        idx: int
    ) -> tuple[
        dict[str, int | NDArray[np.float32]],
        dict[str, bool],
    ]:
        ''' ... '''
        x_imp, _, mask_x, _ = super().__getitem__(idx)
        return x_imp, mask_x
    
    @staticmethod
    def collate_fn(
        batch: list[
            tuple[
                dict[str, int | NDArray[np.float32]],
                dict[str, bool],
            ]
        ]
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        ''' ... '''
        # seperate entries
        x = [smp[0] for smp in batch]
        m = [smp[1] for smp in batch]

        # stack and convert to tensor
        x = {k: torch.as_tensor(np.array([x[i][k] for i in range(len(x))])) for k in x[0]}
        m = {k: torch.as_tensor(np.array([m[i][k] for i in range(len(m))])) for k in m[0]}

        return x, m
    

class TransformerBalancedTrainingDataset(TransformerTrainingDataset):

    def __init__(self,
        src: list[dict[str, Any]], 
        tgt: list[dict[str, Any]],
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        dropout_rate: float = .5,
        dropout_strategy: str = 'compensated',
    ) -> None:
        ''' ... '''
        # call the constructor of parent class
        super().__init__(
            src, tgt, src_modalities, tgt_modalities,
            dropout_rate, dropout_strategy,
        )

        # for each target/label, collect the indices of available cases
        self.tgt_indices: dict[str, dict[int, list[int]]] = dict()
        for tgt_k in self.tgt_modalities:
            tmp = [self.tgt[i][tgt_k] for i in range(len(self.tgt))]
            self.tgt_indices[tgt_k] = dict()
            self.tgt_indices[tgt_k][0] = [i for i in range(len(self.tgt)) if tmp[i] == 0]
            self.tgt_indices[tgt_k][1] = [i for i in range(len(self.tgt)) if tmp[i] == 1]

    def __getitem__(self,
        idx: int
    ) -> tuple[
        dict[str, int | NDArray[np.float32]],
        dict[str, int | NDArray[np.float32]],
        dict[str, bool],
        dict[str, bool],
    ]:
        # select random target, class and index
        tgt_k = random.choice(list(self.tgt_modalities.keys()))
        cls = random.choice([0, 1])
        idx = random.choice(self.tgt_indices[tgt_k][cls])

        # call __getitem__ of super class
        x_imp, y_imp, mask_x, mask_y = super().__getitem__(idx)
        
        # modify mask_y, all targets are masked except tgt_k
        mask_y = {k: [True] for k in self.tgt_modalities}
        mask_y[tgt_k] = [False]

        return x_imp, y_imp, mask_x, mask_y


class Transformer2ndOrderBalancedTrainingDataset(TransformerTrainingDataset):

    def __init__(self,
        src: list[dict[str, Any]], 
        tgt: list[dict[str, Any]],
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        dropout_rate: float = .5,
        dropout_strategy: str = 'compensated',
    ) -> None:
        """ ... """
        # call the constructor of parent class
        super().__init__(
            src, tgt, src_modalities, tgt_modalities,
            dropout_rate, dropout_strategy,
        )

        # construct dictionary of paired tasks
        self.tasks: dict[tuple[str, str], list[int]] = {}
        tgt_keys = list(self.tgt_modalities.keys())
        for tgt_k_0 in tgt_keys:
            for tgt_k_1 in tgt_keys:
                self.tasks[(tgt_k_0, tgt_k_1)] = []

        for i, smp in enumerate(tgt):
            for tgt_k_0 in tgt_keys:
                for tgt_k_1 in tgt_keys:
                    if smp[tgt_k_0] == 0 and smp[tgt_k_1] == 1:
                        self.tasks[(tgt_k_0, tgt_k_1)].append(i)

    def __getitem__(self,
        idx: int
    ) -> tuple[
        dict[str, int | NDArray[np.float32]],
        dict[str, int | NDArray[np.float32]],
        dict[str, bool],
        dict[str, bool],
    ]:
        # select random task
        while True:
            tgt_k_0 = random.choice(list(self.tgt_modalities.keys()))
            tgt_k_1 = random.choice(list(self.tgt_modalities.keys()))
            if len(self.tasks[(tgt_k_0, tgt_k_1)]) != 0:
                idx = random.choice(self.tasks[(tgt_k_0, tgt_k_1)])
                break

        # call __getitem__ of super class
        x_imp, y_imp, mask_x, mask_y = super().__getitem__(idx)
        
        # modify mask_y, all targets are masked except tgt_k
        mask_y = {k: [True] for k in self.tgt_modalities}
        mask_y[tgt_k_0] = [False]
        mask_y[tgt_k_1] = [False]

        return x_imp, y_imp, mask_x, mask_y