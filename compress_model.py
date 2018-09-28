import parameters as para
from Models import segnet, squeezed_segnet, fcn, squeezed_fcn, segnet_depool, tiny_segnet_depool
from custom_metrics import *


def squeeze_weight_matrix(tensor, shrink_ratio=1.0):
    '''To obtain good accuracy using squeezing weights, larger value for shrink_ratio should be used.
    Please note that a shrink_ratio of 1 means the weight matrices will not be shrinked.
    shrink_ratio should be 0.25, 0.5, 0.75 or 1'''
    #print(np.asarray(tensor).shape)
    U, S, Vt = np.linalg.svd(np.asarray(tensor), full_matrices=False)
    print(np.array(U).shape)
    print(np.array(S).shape)
    print(np.array(Vt).shape)
    shrink_factor = int(np.array(S).shape[0] * shrink_ratio)
    #print(shrink_factor)
    squeezed_weight = np.matrix(U[:, :shrink_factor]) * np.diag(S[:shrink_factor]) * np.matrix(Vt[:shrink_factor, :])

    print(np.array(U[:, :shrink_factor]).shape)
    print(np.array(S[:shrink_factor]).shape)
    print(np.array(Vt[:shrink_factor, :]).shape, '\n\n')

    #print(np.asarray(squeezed_weight).shape)
    return np.array(squeezed_weight)


def squeezed_model_SVD(model, skip_layers, s_ratio=1.0):
    squeezed_weights = []
    lr=0
    for i, layers in enumerate(model.layers):
        weights = layers.get_weights()
        if len(np.array(weights).shape) == 2:
            lr += 1
            if lr in skip_layers:
                squeezed_weights.append(weights)
            else:
                # shrink_ratio = 0.25, 0.5, 0.75 or 1
                shrink_weight = squeeze_weight_matrix(weights, shrink_ratio=s_ratio)
                squeezed_weights.append(shrink_weight)
        else:
            squeezed_weights.append(weights)

    for i in range(len(model.layers)):
        model.layers[i].set_weights(squeezed_weights[i])
    #print(model.summary())
    return model


if __name__ == '__main__':
    #   Call model
    #   ==========
    output_rows = 0
    output_cols = 0
    if para.model_type == "segnet":
        print('Initialising Segnet 4 Layers Encoder - Basic ...')
        model, output_rows, output_cols = segnet.SegNet()
    if para.model_type == "squeezed_segnet":
        print('Initialising Squeezed Segnet. Very few parameters ...')
        model, output_rows, output_cols = squeezed_segnet.squeezed_SegNet()
    if para.model_type == "segnet_depool":
        print('Initialising Segnet with Max Pooling Indices...')
        model, output_rows, output_cols = segnet_depool.SegNet()
    if para.model_type == "tiny_segnet_depool":
        print('Initialising Tiny Segnet with Max Pooling Indices...')
        model, output_rows, output_cols = tiny_segnet_depool.SegNet()
    if para.model_type == "fcn":
        print('Initialising FCN...')
        model, output_rows, output_cols = fcn.fcn_8()
    if para.model_type == "squeezed_fcn":
        print('Initialising Squeezed FCN...')
        model, output_rows, output_cols = squeezed_fcn.fcn_8()

    #   Compile the model using sgd optimizer
    #   =====================================
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[dice_coef, precision, recall, 'accuracy'])
    #print(model.summary())

    # Load the model weights
    model.load_weights(para.misc_dir_eval + "/weights.h5")

    #   Compress the Model Weights using Singular Value Decomposition
    #   Don't compress layers 1, 2 and 3. Inportant contextual information held here
    #   =============================================================
    squeezed_model = squeezed_model_SVD(model, skip_layers=[1, 2, 3], s_ratio=para.SVD_ratio)
    squeezed_model.save(para.misc_dir_eval + "/svd_" + str(para.SVD_ratio) + "_weights.h5")
