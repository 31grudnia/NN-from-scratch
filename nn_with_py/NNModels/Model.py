import numpy as np
import pickle
import copy

from . import Layer_Input, Activation_Softmax_Loss_CategoricalCross, Activation_Softmax, Loss_CategoricalCrossentropy

class Model:
    def __init__(self) -> None:
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_num = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_num):
            # If it is first layer, previous is input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and last
            elif i < layer_num - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # Last layer -> next object is loss
            # Save aside the reference to last obj whose output is models output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # if layer contains attribute "weights" its trainable layer
            # add it to list of trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
            # Upadte loss object with trainable layers
            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy
        # create an object of combined activation and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.layers, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCross()
        

    def forward(self, X, training):
        # forward method on Input Layer to set output prop. that
        # the first layer in "prev" object is expecting 
        self.input_layer.forward(X, training)

        # Call forward method for evey obj in chain 
        # Pass output of the previous obj as param 
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    
    def train(self, X, Y, *, epochs=1, batch_size=None, 
              print_every=1, validation_data=None):
        
        self.accuracy.init(Y)
        train_steps = 1
        self.history_train_loss = []
        self.history_train_acc = []
        self.history_test_loss = []
        self.history_test_acc = []

        # Calculate number of steps
        if batch_size is not None:

            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
        
        # Main training loop
        for epoch in range(1, epochs+1):
            print(f"epoch: {epoch}")
            
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            # Iterate over steps
            for step in range(train_steps):
                
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_Y = Y
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size: (step+1)*batch_size]
                    batch_Y = Y[step*batch_size: (step+1)*batch_size]

                output = self.forward(batch_X, training=True)

            
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_Y, 
                                                                     include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_Y)

                self.backward(output, batch_Y)

                # Optimize (update params)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)    
                self.optimizer.post_update_params()
                if not step % print_every or step == train_steps - 1:
                    print(f"step: {step}, " + 
                          f"acc: {accuracy:.3f}, " +
                          f"loss: {loss:.3f} (" +
                          f"data_loss: {data_loss:.3f}, " +
                          f"reg_loss: {regularization_loss:.3f}), "+
                          f"lr: {self.optimizer.current_learning_rate}")
            
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            self.history_train_acc.append(epoch_accuracy)
            self.history_train_loss.append(epoch_loss)

            print(f"training, " + 
                    f"acc: {epoch_accuracy:.3f}, " +
                    f"loss: {epoch_loss:.3f} (" +
                    f"data_loss: {epoch_data_loss:.3f}, " +
                    f"reg_loss: {epoch_regularization_loss:.3f}), "+
                    f"lr: {self.optimizer.current_learning_rate}")
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)
        
        return self.history_train_loss, self.history_train_acc, self.history_test_loss, self.history_test_acc
    

    def evaluate(self, X_val, Y_val, *, batch_size=None):

        validation_steps = 1

        # Calculate num of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add '1' to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            # if, batch size not set - train using one step and full DS
            if batch_size is None:
                batch_X = X_val
                batch_Y = Y_val
            # else, slice batch
            else:
                batch_X = X_val[step*batch_size: (step+1)*batch_size]
                batch_Y = Y_val[step*batch_size: (step+1)*batch_size]
            
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_Y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_Y)
        
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        self.history_test_acc.append(validation_accuracy)
        self.history_test_loss.append(validation_loss)

        print(f"validation, " + 
                f"acc: {validation_accuracy:.3f}, " +
                f"loss: {validation_loss:.3f}")
        

    def predict(self, X, *, batch_size=None):

        # Def value if batch size is not set
        prediction_steps = 1

        # Calculate num of steps
        if batch_size is not None:
            prediction_steps = len(X) //batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add "1' to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []

        for step in range(prediction_steps):
            
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size: (step+1)*batch_size]
            
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)


    def backward(self, output, Y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, Y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]): 
                layer.backward(layer.next.dinputs)
            
            return
        
        self.loss.backward(output, Y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    
    def get_parameters(self):

        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        return parameters
    

    def set_parameters(self, parameters):
        # Iterate over parameters and l;ayers and update layers with each set of params
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)


    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))


    def save(self, path):

        model = copy.deepcopy(self)
        
        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, outputs and dinputs props
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model