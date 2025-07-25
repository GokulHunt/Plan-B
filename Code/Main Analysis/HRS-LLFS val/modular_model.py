import torch
import torch.nn as nn
import torch.optim as optim

# Sub-network for Blue module
class Blue_Module(nn.Module):
    def __init__(self, input_dim):
        super(Blue_Module, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x
    
# Sub-network for Brown module
class Brown_Module(nn.Module):
    def __init__(self, input_dim):
        super(Brown_Module, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Sub-network for Black module
class Black_Module(nn.Module):
    def __init__(self, input_dim):
        super(Black_Module, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 4)
        self.fc3 = nn.Linear(4, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x
    
# Sub-network for Pink module
class Pink_Module(nn.Module):
    def __init__(self, input_dim):
        super(Pink_Module, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 4)
        self.fc3 = nn.Linear(4, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Final combined network
class CombinedNetwork(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, input_dim4):
        super(CombinedNetwork, self).__init__()
        self.module1 = Brown_Module(input_dim1)
        self.module2 = Brown_Module(input_dim2)
        self.module3 = Black_Module(input_dim3)
        self.module4 = Pink_Module(input_dim4)
        
        # Final layer after merging both modules
        self.final_fc1 = nn.Linear(4, 32) 
        self.final_fc2 = nn.Linear(32, 8)
        self.final_fc3 = nn.Linear(8, 1) 
    
    def forward(self, x1, x2, x3, x4):
        out1 = self.module1(x1)
        out2 = self.module2(x2)
        out3 = self.module3(x3)
        out4 = self.module4(x4)
        
        # Concatenating outputs
        combined = torch.cat((out1, out2, out3, out4), dim=1)
        
        # Passing through final layer and applying Sigmoid activation
        output = self.final_fc1(combined)
        output = self.final_fc2(output)
        output = self.final_fc3(output)
        
        return output, out1, out2, out3, out4
    
