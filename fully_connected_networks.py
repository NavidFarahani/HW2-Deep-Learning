"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from libs import Solver

class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # Replace "pass" statement with your code
        
        #    out = w*x + b; so we have:
        input_size=torch.tensor(x.shape[1:])
        x_vector=torch.reshape(x,[x.shape[0],torch.prod(input_size)])
        out=torch.matmul(x_vector,w)+b


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        cache = (x, w, b)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: Implement the linear backward pass.      #
        ##################################################
        # Replace "pass" statement with your code
        
        # We know that out = x*w+ I*b  which I is a ones matrix, so we have:
        # dx =dout*w.T, dw= x.T * dout and db= dout* torch.ones((x.shape[0],1))
        # Note that the dimension of dx is equal to x, dimension of dw is equal to w and dimension of db is equal to b , so we reshaped them.
        
        dx=torch.reshape(torch.matmul(dout,w.T),(x.shape))
        dw=torch.matmul(torch.reshape(x,(x.shape[0],-1)).T,dout)
        db=torch.squeeze(torch.matmul(dout.T,torch.ones((x.shape[0],1),dtype=dx.dtype,device=dx.device)))
        
        ##################################################  
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################

        # We know that for an array, if the value is more than 0, we consider that value,
        # else, we consider zero. So we have:
        the_zeros=torch.zeros(x.shape,dtype=x.dtype,device=x.device)
        out=torch.max(x,the_zeros)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        cache = x

        return out,cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor with an    #
        # in-place operation.                               #
        #####################################################
        # Replace "pass" statement with your code
        
        
        # dx is like a mask, so we determine a ReLU for it with 1 and 0 values and then multiply it
        # to the dout to determine the ReLU function.
        dx=cache
        for i in range(dx.shape[0]):
            for j in range(dx.shape[1]):
                if (dx[i,j]>0):
                    dx[i,j]=1
                else:
                    dx[i,j]=0
                    
        dx=dx*dout
        
        
        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx

class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass (hint: cache = (fc_cache, relu_cache))
        """
        out = None
        cache = None
        ######################################################################
        # TODO: Implement the linear-relu forward pass.                      #
        ######################################################################
        # Replace "pass" statement with your code
        
        # We just have to combine the functions that we wrote by the following code
        # ,so we use linear, then the activation function (or ReLU) after that:    
        out,elements_linear=Linear.forward(x,w,b)
        out,elements_ReLU=ReLU.forward(out)
        cache=(elements_linear,elements_ReLU)
        
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the linear-relu backward pass.                     #
        ######################################################################
        # Replace "pass" statement with your code
        
        # For the backward step, we note that we should act reversely!
        # so we first do backward for ReLU, then we do backward for Linear.
        cache_of_FC,cache_of_ReLU=cache
        ReLU_outs=ReLU.backward(dout,cache_of_ReLU)
        dx,dw,db=Linear.backward(ReLU_outs,cache_of_FC)

        ######################################################################
        #                END OF YOUR CODE                                    #
        ######################################################################
        return dx, dw, db


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss = None
    dx = None
    ######################################################################
    # TODO: Implement the Softmax layer.                                 #
    ######################################################################
    # Replace "pass" statement with your code

    #We could use this program, but it is very slow
    
    
    # dx=torch.zeros(x.shape)

    # for i in range(x.shape[0]):
    #     max_per_row=torch.max(x[i,:])
    #     out_soft_max_per_row=np.exp(x[i,:]-max_per_row)
    #     dx[i,:]=out_soft_max_per_row/torch.sum(out_soft_max_per_row)


    # So we use this code
    
    # calculation of e^(x-max(x)) for all rows and then, normalize them due
    # to sum of all rows of dx be equal to 1.
    
    max_per_row=torch.max(x,dim=1,keepdim=True)[0]
    dx=torch.exp(x-max_per_row)
    sum_per_row=torch.sum(dx,dim=1,keepdim=True)
    dx=dx/sum_per_row

    #Computation of loss for sigmoid
    
    # we have cross entropy function, so we have:
    loss=0
    for j in range(x.shape[0]):
        torch.log(dx[j,y[j]])
        loss=loss-torch.log(dx[j,y[j]])
    loss=loss/x.shape[0]

    
    
    #For gradient of the softmax, we use this code to calculate, differentials:
    for k in range(x.shape[0]):
        dx[k,y[k]]=dx[k,y[k]]-1
    
    dx=dx/x.shape[0]

    
    ######################################################################
    #                END OF YOUR CODE                                    #
    ######################################################################
    return loss, dx

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # Replace "pass" statement with your code
        
        #You said that biases should be initialized to zero, so I used torch.zeros
        w1=torch.normal(0,weight_scale,dtype=dtype,size=(input_dim,hidden_dim))
        b1=torch.zeros(hidden_dim,dtype=dtype,device=device)
        
        w2=torch.normal(0,weight_scale,dtype=dtype,size=(hidden_dim,num_classes))
        b2=torch.zeros(num_classes,dtype=dtype,device=device)
        
        self.params={"W1":w1,"b1":b1,"W2":w2,"b2":b2}
        
        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        # Replace "pass" statement with your code
        
        # We just use our written functions here with w and b parameters from this class.
        
        out,elements_ReLU=Linear_ReLU.forward(X,self.params["W1"],self.params["b1"])
        scores,elements_Linear=Linear.forward(out,self.params['W2'],self.params['b2'])
        
        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax, and make sure that #
        # grads[k] holds the gradients for self.params[k]. Don't forget   #
        # to add L2 regularization!                                       #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and       #
        # you pass the automated tests, make sure that your L2            #
        # regularization does not include a factor of 0.5.                #
        ###################################################################
        # Replace "pass" statement with your code
        loss,dout=softmax_loss(scores,y)
        
        dout,grads['W2'],grads['b2']=Linear.backward(dout,elements_Linear)
        
        #L2 regularization does not include 0.5, so, derivative of (||w||^2 is 2*w so we have):   
        grads['W2']=grads['W2']+2*self.reg*self.params['W2']
        #grads['b2']=grads['b2']+2*self.reg*self.params['b2']
        
        dx,grads['W1'],grads['b1']=Linear_ReLU.backward(dout,elements_ReLU)
        
        #L2 regularization does not include 0.5, so, derivative of (||w||^2 is 2*w so we have):
        grads['W1']=grads['W1']+2*self.reg*self.params['W1']
        #grads['b1']=grads['b1']+2*self.reg*self.params['b1']
        
        
        loss=loss+self.reg*(torch.sum(self.params['W1']**2)+torch.sum(self.params['W2']**2))
        
        
        
        #It seems that we should not use regularization for biases, because we will encounter loss!
        # So, I did not calculate regularization for biases and I calculated loss with just considering w as regularization term.
        
        #loss=loss+self.reg*(torch.sum(self.params['W1']**2)+torch.sum(self.params['W2']**2)+torch.sum(self.params['b1']**2)+torch.sum(self.params['b2']**2))
        
        
        
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss,grads

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all         #
        # values in the self.params dictionary. Store weights and biases      #
        # for the first layer in W1 and b1; for the second layer use W2 and   #
        # b2, etc. Weights should be initialized from a normal distribution   #
        # centered at 0 with standard deviation equal to weight_scale. Biases #
        # should be initialized to zero.                                      #
        #######################################################################
        # Replace "pass" statement with your code

        
        dimension_of_layer=[input_dim,*hidden_dims,num_classes]
        #You said that biases should be initialized to zero, so I used torch.zeros
        # Note that for layer (i+1), we have to consider dimension_of_layer[i+1] because the first layer is input
        # so we added 1 to i .
        for i in range(self.num_layers):
             w=torch.normal(0,weight_scale,(dimension_of_layer[i],dimension_of_layer[i+1]), dtype = dtype,device=device)
             self.params[f'W{i+1}']=w
             b=torch.zeros(dimension_of_layer[i+1], dtype = dtype,device=device)
             self.params[f'b{i+1}']=b
        
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ##################################################################
        # TODO: Implement the forward pass for the fully-connected net,  #
        # computing the class scores for X and storing them in the       #
        # scores variable.                                               #
        #                                                                #
        # When using dropout, you'll need to pass self.dropout_param     #
        # to each dropout forward pass.                                  #
        ##################################################################
        # Replace "pass" statement with your code
        
        scores=X
        caches=[]
        # For all layers, we use ReLU but for the last layer, we do not use ReLU, so we have:
        for i in range(self.num_layers):
            if(i<self.num_layers-1):
                scores,cache=Linear_ReLU.forward(scores,self.params[f'W{i+1}'],self.params[f'b{i+1}'])
            else:
                scores,cache=Linear.forward(scores,self.params[f'W{i+1}'],self.params[f'b{i+1}'])
            caches.append(cache)

        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: Implement the backward pass for the fully-connected net.    #
        # Store the loss in the loss variable and gradients in the grads    #
        # dictionary. Compute data loss using softmax, and make sure that   #
        # grads[k] holds the gradients for self.params[k]. Don't forget to  #
        # add L2 regularization!                                            #
        # NOTE: To ensure that your implementation matches ours and you     #
        # pass the automated tests, make sure that your L2 regularization   #
        # includes a factor of 0.5 to simplify the expression for           #
        # the gradient.                                                     #
        #####################################################################
        # Replace "pass" statement with your code
        
        # Because of regularization, we have to use this code for all layers
        total_loss=0
        for i in range(self.num_layers):
            total_loss=total_loss + (self.reg) *torch.sum(self.params[f'W{i+1}']**2)
        
        # Moreover, for the last layer we should add softmax_loss to all losses.
        loss_softmax,dout=softmax_loss(scores,y)
        total_loss=total_loss+loss_softmax

        loss=total_loss
        
        
        # For all layers, we should pass the backward way for all calculations, so we start from num_layers,
        # to 0 with step = -1.
        # Note that we have regularization, so we should for all layers calculate dw with considering the self.reg term.
        # but we do not have to calculate regularization for db!
        
        for i in range( (self.num_layers)-1,-1,-1 ):
            cache=caches[i]
            
            if(i<self.num_layers-1):
                dout,dw,db=Linear_ReLU.backward(dout,cache)
            else:
                dout,dw,db=Linear.backward(dout,cache)
            
            dw=dw+2*self.reg*self.params[f'W{i+1}']
            grads[f'W{i+1}']=dw
            grads[f'b{i+1}']=db
            
        ###########################################################
        #                   END OF YOUR CODE                      #
        ###########################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    #############################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that   #
    # achieves at least 50% accuracy on the validation set.     #
    #############################################################
    solver = None
    # Replace "pass" statement with your code
    
    #use Solver from libs!!
    
    solver=Solver(model,data_dict,device=device)
    ##############################################################
    #                    END OF YOUR CODE                        #
    ##############################################################
    return solver


def get_three_layer_network_params():
    ###############################################################
    # TODO: Change weight_scale and learning_rate so your         #
    # model achieves 100% training accuracy within 20 epochs.     #
    ###############################################################
    
    # I adjucstet these parameters
    
    weight_scale = 0.15   # Experiment with this!
    learning_rate = 0.6  # Experiment with this!
    ################################################################
    #                             END OF YOUR CODE                 #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    
    # I adjucstet these parameters
    
    
    learning_rate = 0.6 # Experiment with this!
    weight_scale = 0.1   # Experiment with this!
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate



def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    # Replace "pass" statement with your code
    
    # We know that for SGD, we have to consider momentum to the gradient, so we have:
    v=config['momentum']*v-config['learning_rate']*dw
    next_w=w+v
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity']=v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # Replace "pass" statement with your code
    
    # for rmsprop, I used the link that you refernced to it and the algorithm is:
    
    conf=config['decay_rate']*config['cache']+(1-config['decay_rate'])*dw**2
    w=w-( config['learning_rate']*dw / ( torch.sqrt(conf) + config['epsilon'] ) )
    
    config['cache']=conf
    next_w=w
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in#
    # the next_w variable. Don't forget to update the m, v, and t variables  #
    # stored in config.                                                      #
    #                                                                        #
    # NOTE: In order to match the reference output, please modify t _before_ #
    # using it in any calculations.                                          #
    ##########################################################################
    # Replace "pass" statement with your code
    
    # for adam, I used the link that you refernced to it and the algorithm is:
    
    config['m']=config['beta1']*config['m']+(1-config['beta1'])*dw
    config['t']=config['t']+1
    
    m_t=config['m']/(1-config['beta1']**config['t'])
    config['v']=config['beta2']*config['v']+(1-config['beta2'])*(dw**2)
    v_t=config['v']/(1-config['beta2']**config['t'])
    w=w-config['learning_rate']*  ( m_t/(  torch.sqrt(v_t)+config['epsilon']  ) )
    
    next_w=w
    #########################################################################
    #                              END OF YOUR CODE                         #
    #########################################################################

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # Replace "pass" statement with your code
            
            #for dropout, we consider a musk that will multiply to our inputs
            # we have to eliminate some of them, so we consider a uniform random between 0 and 1
            # and if the value is less than p we consider it 0 and eliminate that x, and if
            # it is more than p, we do not eliminate it and also we increase it to (1/(1-p))  more than
            # its previous value,
            mask=torch.rand(x.shape)
            mask[mask<p]=0
            mask[mask>p]=1/(1-p)
            
            out=mask*x

            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            # Replace "pass" statement with your code
            
            #For test, we do not have dropout, because dropout process is just for training step to reduce the computation complexity
            
            
            out=x
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # Replace "pass" statement with your code
            
            #For training step, we have dropout, because dropout process is for training step to reduce the computation complexity
            # by using mask matrix.
            
            dx=dout*mask
            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx

