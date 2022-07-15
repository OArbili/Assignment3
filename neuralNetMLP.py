import numpy as np
import utils
##########################
### MODEL
##########################

class NeuralNetMLP:

    def __init__(self, num_features, num_layers, num_hidden_l1, num_hidden_l2,num_classes, min_batch_size=100, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        self.min_batch_size = min_batch_size
        # hidden
        rng = np.random.RandomState(random_seed)
        if num_layers > 2:
            print('This NN class produce up to 2 layers Neural network')
            assert(num_layers<=2)
        else:
            self.num_layers = num_layers

            self.weight_h_1 = rng.normal(
                loc=0.0, scale=0.1, size=(num_hidden_l1, num_features))
            self.bias_h_1 = np.zeros(num_hidden_l1)
            
            # two layers init
            if (num_layers>1):
                self.weight_h_2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden_l2, num_hidden_l1))
                self.bias_h_2 = np.zeros(num_hidden_l2)
                
                # output
                self.weight_out = rng.normal(
                    loc=0.0, scale=0.1, size=(num_classes, num_hidden_l2))
                self.bias_out = np.zeros(num_classes)
                
            else:
                # output
                self.weight_out = rng.normal(
                    loc=0.0, scale=0.1, size=(num_classes, num_hidden_l1))
                self.bias_out = np.zeros(num_classes)
        
    def forward_one_layer(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h_1.T) + self.bias_h_1
        a_h = utils.sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = utils.sigmoid(z_out)
        return a_h, a_out

    # todo change names and logics
    def forward_two_layers(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h_1.T) + self.bias_h_1        
        a_h = utils.sigmoid(z_h)
        # Hidden layer 2
        # input dim: [n_examples, n_hidden] dot [n_hidden_2, n_hidden].T
        # output dim: [n_examples, n_hidden_2]
        z_h_2 = np.dot(a_h, self.weight_h_2.T) + self.bias_h_2        
        a_h_2 = utils.sigmoid(z_h_2)

        # Output layer
        # input dim: [n_examples, n_hidden_2] dot [n_classes, n_hidden_2].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h_2, self.weight_out.T) + self.bias_out
        a_out = utils.sigmoid(z_out)
        return a_h, a_h_2, a_out
    
    # x input for the model (100,28*28)
    # a_h - first hidden layer after sigmoid (range 0-1)
    # a_out - output layer (prediction) with range(0-1) return probability
    # y - true label (int)
    def backward_one_layer(self, x, a_h, a_out, y):
    
        #########################
        ### Output layer weights
        #########################
        
        # onehot encoding
        y_onehot = utils.int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        # Ofir - calc the drivatives of the error rmse (?)
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # Ofir sigmoid derivatives
        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # ofir - chain rule
        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # "delta (rule) placeholder"

        # gradient for output weights
        
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        

        #################################        
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # ofir - self.weight_out is the prevoius calcualtion of d_loss__dw_out
        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out
        
        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative
        
        # [n_examples, n_features]
        d_z_h__d_w_h = x
        
        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h, d_loss__d_b_h)
    
    # todo change names and logics
    def backward_two_layers(self, x, a_h_1, a_h_2, a_out, y):  
    
        #########################
        ### Output layer weights
        #########################
        
        # onehot encoding
        y_onehot = utils.int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # "delta (rule) placeholder"

        # gradient for output weights
        
        # [n_examples, n_hidden_2]
        d_z_out__dw_out = a_h_2
        
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden_2]
        # output dim: [n_classes, n_hidden_2]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)      
        
        #################################        
        # Part 2: dLoss/dHiddenWeights_2 = DeltaOut * dOutNet/dHiddenAct_2 * dHiddenAct_2/dHiddenNet_2 * dHiddenNet_2/dWeight_2
        # where DeltaHidden2 = dOutNet/dHiddenAct_2 * dHiddenAct_2/dHiddenNet_2
        
        # [n_classes, n_hidden_2]
        d_z_out__a_h_2 = self.weight_out
        
        # output dim: [n_examples, n_hidden_2]
        d_loss__a_h_2 = np.dot(delta_out, d_z_out__a_h_2)
        
        # [n_examples, n_hidden_2]
        d_a_h_2__d_z_h_2 = a_h_2 * (1. - a_h_2) # sigmoid derivative
        
        delta_hidden_2 = d_loss__a_h_2 * d_a_h_2__d_z_h_2
        
        # [n_examples, n_features]
        d_z_h_2__d_w_h_2 = a_h_1
        # output dim: [n_hidden_2, n_features]
        d_loss__d_w_h_2 = np.dot((delta_hidden_2).T, d_z_h_2__d_w_h_2)
        d_loss__d_b_h_2 = np.sum((delta_hidden_2), axis=0)

        #################################        
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * DeltaHidden2 * dHiddenNet_2/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # [n_classes, n_hidden]
        d_z_h_2__a_h_1 = self.weight_h_2
        
        # output dim: [n_examples, n_hidden]
        d_loss__a_h_1 = np.dot(delta_hidden_2, d_z_h_2__a_h_1)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h_1 = a_h_1 * (1. - a_h_1) # sigmoid derivative
        
        # [n_examples, n_features]
        d_z_h__d_w_h_1 = x
        
        # output dim: [n_hidden, n_features]
        d_loss__d_w_h_1 = np.dot((d_loss__a_h_1 * d_a_h__d_z_h_1).T, d_z_h__d_w_h_1)
        d_loss__d_b_h_1 = np.sum((d_loss__a_h_1 * d_a_h__d_z_h_1), axis=0)        

        return (d_loss__dw_out,  d_loss__db_out, 
                d_loss__d_w_h_2, d_loss__d_b_h_2,
                d_loss__d_w_h_1,   d_loss__d_b_h_1
               )

    def train(self, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    
        epoch_loss = []
        epoch_train_acc = []
        epoch_valid_acc = []
        
        for e in range(num_epochs):

            # iterate over minibatches
            minibatch_gen = utils.minibatch_generator(
                X_train, y_train, self.min_batch_size)

            for X_train_mini, y_train_mini in minibatch_gen:
                
                #### Compute gradients ####
                if (self.num_layers==1):
                    a_h, a_out = self.forward_one_layer(X_train_mini)
                    d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                        self.backward_one_layer(X_train_mini, a_h, a_out, y_train_mini)

                    #### Update weights ####
                    self.weight_h_1 -= learning_rate * d_loss__d_w_h
                    self.bias_h_1 -= learning_rate * d_loss__d_b_h
                    self.weight_out -= learning_rate * d_loss__d_w_out
                    self.bias_out -= learning_rate * d_loss__d_b_out
                else:
                    #### Compute outputs ####
                    a_h, a_h_2, a_out = self.forward_two_layers(X_train_mini)

                    #### Compute gradients ####
                    d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h_2, d_loss__d_b_h_2, d_loss__d_w_h_1, d_loss__d_b_h_1  = \
                        self.backward_two_layers(X_train_mini, a_h, a_h_2, a_out, y_train_mini)

                    #### Update weights ####            
                    self.weight_h_1 -= learning_rate * d_loss__d_w_h_1
                    self.bias_h_1 -= learning_rate * d_loss__d_b_h_1

                    self.weight_h_2 -= learning_rate * d_loss__d_w_h_2
                    self.bias_h_2 -= learning_rate * d_loss__d_b_h_2
                    self.weight_out -= learning_rate * d_loss__d_w_out
                    self.bias_out -= learning_rate * d_loss__d_b_out
            
            #### Epoch Logging ####        
            train_mse, train_acc = utils.compute_mse_and_acc(self, X_train, y_train)
            valid_mse, valid_acc = utils.compute_mse_and_acc(self, X_valid, y_valid)
            train_acc, valid_acc = train_acc*100, valid_acc*100
            epoch_train_acc.append(train_acc)
            epoch_valid_acc.append(valid_acc)
            epoch_loss.append(train_mse)
            print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
                f'| Train MSE: {train_mse:.2f} '
                f'| Train Acc: {train_acc:.2f}% '
                f'| Valid Acc: {valid_acc:.2f}%')

        return epoch_loss, epoch_train_acc, epoch_valid_acc
    
    def predict(self,X, num_labels=10, ret_proba=False):
        if (self.num_layers==1):
            _, probas = self.forward_one_layer(X)
        else:
            _, _, probas = self.forward_two_layers(X)

        if ret_proba:
            return probas
        else:
            return np.argmax(probas, axis=1)
