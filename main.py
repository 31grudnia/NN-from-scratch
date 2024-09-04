import numpy as np 
import nnfs

from nn_with_py.ActivationFunctions.ActivationSoftamxWLossCategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCross
from nn_with_py.ActivationFunctions.ActivationSoftmax import Activation_Softmax
from nn_with_py.ActivationFunctions.ActivationRelu import Activation_Relu
from nn_with_py.LossFunctions.LossCategoricalCrossentropy import Loss_CategoricalCrossentropy

from nn_with_py.Layers.LayerDense import Layer_Dense

from nn_with_py.utils.data_setup import load_files, fashion_minst_download

if __name__ == "__main__":
    
    path = "/Users/mikolajstarczewski/Desktop/Magisterka/NN_with_py/nn_with_py/data_fiiles/"
    signs, labels = load_files(path)
    
    print(f"")
    print("SIGNS")
    print(f"Type: {type(signs)}")
    print(f"Shape: {signs.shape}")

    print("LABELS")
    print(f"Type: {type(labels)}")
    print(f"Shape: {labels.shape}")
    print(f"Characters index: {labels[33390]}")
    print(len(labels))

    # print("SIGNS DICT aka JSON")
    # print(f"Type: {type(signs_dictionary)}")
    # print(f"Length: {len(signs_dictionary)}")
    # print(f"Character: {signs_dictionary[13]}")

    # softmax_outputs = np.array([[0.7, 0.1, 0.2],
    #                            [0.1, 0.5, 0.4],
    #                            [0.02, 0.9, 0.08]])
    
    # class_targets = np.array([0, 1, 1])
    # softmax_loss = Activation_Softmax_Loss_CategoricalCross()
    # softmax_loss.backward(softmax_outputs, class_targets)
    # dvalues1 = softmax_loss.dinputs

    # activation = Activation_Softmax()
    # activation.output = softmax_outputs
    # loss = Loss_CategoricalCrossentropy()
    # loss.backward(softmax_outputs, class_targets) 
    # activation.backward(loss.dinputs)
    # dvalues2 = activation.dinputs

    # print(f"Gradients: combined loss and actv: {dvalues1}")
    # print(f"Gradients: separate loss and actv: {dvalues2}")