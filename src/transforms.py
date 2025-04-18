"""Transformations for data processing."""
import torch as th


class Transform:
    """Base class for transforms."""
    def transform(self, tensor):
        """Apply transformation to input tensor."""
        raise NotImplementedError
    
    def infer_output_info(self, vshape_in, dtype_in):
        """Infer output shape and type from input."""
        raise NotImplementedError


class OneHot(Transform):
    """One-hot encoding transform."""
    def __init__(self, out_dim):
        self.out_dim = out_dim
    
    def transform(self, tensor):
        """Transform input tensor to one-hot encoded format."""
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()
    
    def infer_output_info(self, vshape_in, dtype_in):
        """Infer output shape and type."""
        return (self.out_dim,), th.float32