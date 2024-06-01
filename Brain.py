import torch.nn as nn

class NeuralNet(nn.Module):
    
    
     #self: It represents the instance of the class.    
     # input_size: The number of input features or dimensions.
     # hidden_size: The number of neurons in the hidden layer.
     # num_classes: The number of output classes.
        
      def __inti__(self,input_size,hidden_size,num_classes):                                           
         super(NeuralNet,self).__init__()
         self.l1 = nn.Linear(input_size, hidden_size)
         self.l2 = nn.Linear(hidden_size, hidden_size)
         self.l3 = nn.Linear(hidden_size,num_classes)
         self.relu = nn.ReLU()

      def forward(self,x):
         out = self.l1(x)
         out = self.relu(out)
         out = self.l2(out)
         out = self.relu(out)   #This line creates an instance of the ReLU activation function (nn.ReLU()), which will be used to introduce non-linearity after each linear layer.
         out = self.l3(out)     #This line creates the third linear transformation layer, mapping the hidden layer to the output layer with num_classes output units.
         return out
      
        
        
    
    
    
    