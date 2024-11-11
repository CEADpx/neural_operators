# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import numpy as np
import scipy

class SaveData:
    def __init__(self):
        pass

    def save(self, sess, X_trunk_pos, fnn_model, \
             W_T, b_T, W_B, b_B, \
             X_trunk_min, X_trunk_max, \
             X_ph, Y_ph, \
             data, num_test, \
             num_tr_outputs, num_Y_components, \
             results_dir, data_prefix):
        
        X_test, X_trunk_test, Y_test = data.testbatch(num_test)
        X_trunk = tf.tile(X_trunk_pos[None, :, :], [num_test, 1, 1])

        test_dict = {X_ph: X_test, Y_ph: Y_test}

        num_out_fn_points = X_trunk_test.shape[0]

        # trunk net
        u_T = fnn_model.fnn_T(W_T, b_T, X_trunk, X_trunk_min, X_trunk_max)
        
        # branch net
        u_B = fnn_model.fnn_B(W_B, b_B, X_ph)
        u_B = tf.tile(u_B, [1, X_trunk_test.shape[0], 1])

        # inner product
        if data_prefix == 'LinearElasticity':
            Y_pred = [u_B[:, :, :num_tr_outputs]*u_T, u_B[:, :, num_tr_outputs:]*u_T]
            Y_pred = tf.stack(Y_pred, axis=-1)
            Y_pred = tf.reduce_sum(Y_pred, axis=2)
            Y_pred = tf.reshape(Y_pred, [-1, num_out_fn_points*num_Y_components, 1], name='F')
        else:
            Y_pred = u_B*u_T
            Y_pred = tf.reduce_sum(Y_pred, axis=-1, keepdims=True)

        # compute predictions
        Y_pred_ = sess.run(Y_pred, feed_dict=test_dict)

        # compute relative L2 error
        err = np.mean(np.square(Y_pred_ - Y_test))
        print('error (l2 squared): %.3f'%(err))

        err = np.reshape(err, (-1, 1))
        
        np.savetxt(results_dir+'/err.txt', err, fmt='%e')
        
        scipy.io.savemat(results_dir + data_prefix + '_DeepONet.mat', 
                     mdict={'X_test': X_test,
                            'Y_test': Y_test, 
                            'Y_pred': Y_pred_})
        