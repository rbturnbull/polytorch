from typing import List

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import plotly.graph_objects as go

from .data import PolyData
from .util import permute_feature_axis

class ContinuousEmbedding(nn.Module):
    def __init__(
        self, 
        embedding_size:int,
        bias:bool=True,
        mean:float|None=None,
        stdev:float|None=None,
        device=None, 
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)    
        
        self.embedding_size = embedding_size
        self.mean = mean
        self.stdev = stdev

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((embedding_size,), **factory_kwargs), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty((embedding_size,), **factory_kwargs), requires_grad=True)
        else:
            self.bias = Parameter(torch.zeros((embedding_size,), **factory_kwargs), requires_grad=False)

        self.reset_parameters()

    def forward(self, input):
        x = input.flatten().unsqueeze(1)
        if self.mean is not None:
            x = x - self.mean
        if self.stdev is not None:
            x = x/self.stdev
        embedded = self.bias + x * self.weight.unsqueeze(0)
        embedded = embedded.reshape(input.shape + (-1,))

        return embedded

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        torch.nn.init.constant_(self.bias, 0.0)


class OrdinalEmbedding(ContinuousEmbedding):
    def __init__(
        self,
        category_count,
        embedding_size,
        bias:bool=True,
        device=None, 
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            embedding_size,
            bias=bias,
            device=device, 
            dtype=dtype,
            **kwargs,
        )    
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.distance_scores = Parameter(torch.ones((category_count-1,), **factory_kwargs), requires_grad=True)

    def forward(self, x):
        distances = torch.cumsum(F.softmax(self.distance_scores, dim=0), dim=0)
        
        # prepend zero
        distances = torch.cat([torch.zeros((1,), device=distances.device, dtype=distances.dtype), distances])
        distance = torch.gather(distances, 0, x.flatten())
        embedded = self.bias + distance.unsqueeze(1) * self.weight.unsqueeze(0)
        embedded = embedded.reshape(x.shape + (-1,))

        return embedded


class PolyEmbedding(nn.Module):
    def __init__(
        self,
        input_types:List[PolyData],
        embedding_size:int,  
        feature_axis:int=-1,      
        **kwargs,
    ):
        super().__init__(**kwargs)    
        self.input_types = input_types
        self.embedding_size = embedding_size
        self.embedding_modules = nn.ModuleList([
            input.embedding_module(embedding_size) for input in input_types
        ])
        self.feature_axis = feature_axis

    def forward(self, *inputs):
        shape = inputs[0].shape + (self.embedding_size,)
        embedded = torch.zeros( shape, device=inputs[0].device ) 

        for input, module in zip(inputs, self.embedding_modules):
            if input.dtype == torch.bool:
                input = input.int()
            embedded += module(input)

        return permute_feature_axis(embedded, old_axis=-1, new_axis=self.feature_axis)

    def plot(self, **kwargs) -> go.Figure:
        from .plots import plot_embedding
        return plot_embedding(self, **kwargs)

