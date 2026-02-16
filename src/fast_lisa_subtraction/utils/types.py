import torch
import pandas as pd
from tensordict import TensorDict

#=======================================
#TensorDict Wrapper Class
#=======================================
class TensorSamples(TensorDict):
    """Wrapper class for TensorDict to better manage samples"""
    
    def flatten(self):
        """Returns the samples as a single flattened tensor"""
        return self.to_tensor().flatten()
    
    def to_tensor(self):
        """Returns the samples as a single tensor"""
        return torch.stack([self[key] for key in self.keys()], dim=-1)

    def tensor(self):
        """Returns the samples as a single tensor"""
        return self.to_tensor()
    
    def numpy(self):
        """Returns the samples as a numpy array"""
        return self.tensor().cpu().numpy()
    
    def dict(self):
        """Returns the samples as a dict of tensors"""
        return self.to_dict()

    def numpy_dict(self):
        """Returns the samples as a dict of numpy arrays"""
        return {key: self[key].cpu().numpy() for key in self.keys()}
    
    def dataframe(self):
        """Returns the samples as a pandas DataFrame"""
        return pd.DataFrame(self.numpy_dict())
