import numpy as np

# @TODO
# some kind of abstract
# does it need tests?
# idk
class Loss:
    
    # def __init__(self):
    #     self.accumulated_sum = 0
    #     self.accumulated_count = 0
    
    def regularization_loss(self):
        regularization_loss = 0
        
        # Calculate regularization loss iterate all trainable layers
        for layer in self.trainable_layers:
            
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))
                
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights * layer.weights)
            
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases * layer.biases)

        return regularization_loss
    

    def remember_trainabler_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    def calculate(self, output, y, *, include_regularization=True):
        sample_losses = self.forwad(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    

    def calculate_accumulated(self, *, include_regularization=True):
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    