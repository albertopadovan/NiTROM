import torch

class full_order_model:
        
    def __init__(self,C):
        
        self.C = C      # The output matrix

    def compute_output(self,q):
        return torch.matmul(self.C,q)
    
    def compute_output_derivative(self,q):
        return self.C
    
class full_order_model_identity:
        
    def __init__(self):
        pass

    def compute_output(self,q):
        return q