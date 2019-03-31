import os
import os.path as osp
import argparse

import tensorflow as tf

from keras.models import load_model
from keras import backend as K
import parameters as para
from Models import fcn, squeezed_fcn, segnet_depool, tiny_segnet_depool, sub_pixel, tiny_sub_pixel



def keras2Tensorflow(model_path=para.misc_dir_eval + "/weights.h5", model_path_pb=para.misc_dir_eval, model_outputs=1,
                     K2T_prefix="output_node", model_file_pb="model.pb"):
    '''
        model_path = The HDF5 Keras model you wish to convert to .pb
        model_outputs = The number of outputs in the HDF5 model
        model_path_pb = The directory to place the output .pb files for tensorflow
        K2T_prefix = "k2tfout"
        model_file_pb = "model.pb" #
    '''

    #   Call model
    #   ==========
    output_rows = 0
    output_cols = 0
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
    if para.model_type == "sub_pixel":
        print('Initialising sub_pixel ...')
        model, output_rows, output_cols = sub_pixel.subPixelModel()
    if para.model_type == "tiny_sub_pixel":
        print('Initialising tiny_sub_pixel ...')
        model, output_rows, output_cols = tiny_sub_pixel.tinySubPixelModel()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    if not os.path.isdir(model_path_pb):
        os.mkdir(model_path_pb)

    K.set_learning_phase(0)

    model.load_weights(para.misc_dir_eval + "/weights.h5")
    #net_model = load_model(model_path)

    # Alias the outputs in the model - this sometimes makes them easier to access in TF
    pred = [None] * model_outputs
    pred_node_names = [None] * model_outputs
    for i in range(model_outputs):
        pred_node_names[i] = K2T_prefix + str(i)
        pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
    print('Output nodes names are: ', pred_node_names)

    sess = K.get_session()

    # Write the graph in human readable
    f = 'graph_def_for_reference.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), model_path_pb, f, as_text=True)
    print('Saved the graph definition in ascii format at: ', osp.join(model_path_pb, f))

    # Write the graph in binary .pb file
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, model_path_pb, model_file_pb, as_text=False)
    print('Saved the constant graph (ready for inference) at: ', osp.join(model_path_pb, model_file_pb))


if __name__ == '__main__':
    keras2Tensorflow()
