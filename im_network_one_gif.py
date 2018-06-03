import tensorflow as tf
from DNN_v2 import *
from config import cfg
import math
import utils


def euclidean(x, y, multi=100.0):
    """
      x:      [0,0],[1,1],[2,2]
      y:      [3,4 ],[7,9],[5,5]
      output: [  5.         10.          4.2426405]
    """
    multiplier = tf.constant(multi, dtype='float') #for bc #10000
    sub = x * multiplier - y * multiplier
    
    # mean square error with multiplier
    return tf.reduce_mean(tf.square(sub))
    # return tf.sqrt(tf.reduce_sum(tf.square(x-y),1))


def print_each_w_b():
    sort_net_key = sorted(cfg['network'])  
    for key in sort_net_key:
        com = cfg['network'][key] 
        print('\t key:' + key)
        print('\t\t w:' + str(com['w']) )
        print('\t\t b:' + str(com['b']) )
        

class BehaviorClone(object):
    def __init__(self, training=True, logger=None):
        self.logger = logger if logger is not None else utils.set_logger(['bulid'], 'bulid_log')[1]
        self.batch_size = cfg['batch_size']
        self.pics_each_gif = cfg['pics_each_gif']
        self.img_w = cfg['image_width']
        self.img_h = cfg['image_height']
        self.ext_w = cfg['extra_width']
        self.ext_h = cfg['extra_height']
        self.img_d = cfg['image_depth']
        self.feedback_num = cfg['robot_feedback_num']
        self.outs = cfg['network_con']['im_prediction']['size']
        self.training = training
        self.set_network_property()
        
        with tf.variable_scope("tower"): 
            self.build_weight()

        # self.build_inputs_and_outputs()
        # with tf.Graph().as_default() as graph:
        #     self.graph = graph
            # self.saver = tf.train.Saver()

    def build_weight(self):
        # because yaml load cannot load by order, use sorted to sort it,
        # very tricky, becarefully
        fc_down_factor1 = 0
        pre_com = None
        net_name = 'network'
        for key in sorted(cfg[net_name]):
            com = cfg[net_name][key]        #component
            
            if com['type'] == 'conv':
                if pre_com == None:
                    in_channel = self.img_d
                elif pre_com['type'] == 'conv':
                    in_channel = pre_com['out_channel']
                else:
                    self.logger.error('build_weight() say Error component property, in conv else')

                w_shape = [com['kernel_size'], com['kernel_size'], in_channel, com['out_channel']]
                com['w'] = weight_variable(w_shape , name= net_name + '_' + key + "_w") 
                com['b'] = bias_variable([com['out_channel']]  , name= net_name + '_' + key + "_b")
                fc_down_factor1 = fc_down_factor1 * int(com['stride']) if fc_down_factor1 is not 0 else int(com['stride'])
            pre_com = com

        fc_down_factor2 = 0
        pre_com1 = None
        net_name = 'network_ext'
        for key in sorted(cfg[net_name]):
            com = cfg[net_name][key]        #component
            
            if com['type'] == 'conv':
                if pre_com1 == None:
                    in_channel = self.img_d
                elif pre_com1['type'] == 'conv':
                    in_channel = pre_com1['out_channel']
                else:
                    self.logger.error('build_weight() say Error component property, in conv else')

                w_shape = [com['kernel_size'], com['kernel_size'], in_channel, com['out_channel']]
                com['w'] = weight_variable(w_shape , name= net_name + '_' + key + "_w") 
                com['b'] = bias_variable([com['out_channel']]  , name= net_name + '_' + key + "_b")
                fc_down_factor2 = fc_down_factor2 * int(com['stride']) if fc_down_factor2 is not 0 else int(com['stride'])
            pre_com1 = com

        net_name = 'network_con'
        for key in sorted(cfg[net_name]):
            com = cfg[net_name][key]        #component

            if  com['type'] == 'fc':
                if pre_com['type'] == 'conv':
                    # if 'spatial_softmax' in pre_com:
                    #     in_channel = self.feedback_num + int(pre_com['out_channel'] * 2)
                    # else:
                    #     in_channel = self.feedback_num + math.ceil(self.img_w/ fc_down_factor) * math.ceil(self.img_h/ fc_down_factor) * pre_com['out_channel'] #because default padding is 'SAME'
                    in_channel = int(pre_com['out_channel'] * 2) + \
                        + math.ceil(cfg['extra_width']/ fc_down_factor2) * math.ceil(cfg['extra_height']/ fc_down_factor2) * pre_com1['out_channel']

                    self.logger.info('first layer')
                    self.logger.info('\t fc_down_factor = '+ str(fc_down_factor2))
                    self.logger.info('\t in_channel = '+ str(in_channel))
                elif key == 'im_fc_2':
                    in_channel = self.feedback_num
                elif key == 'im_fc_3':
                    in_channel = cfg[net_name]['im_fc_1']['size'] + cfg[net_name]['im_fc_2']['size']
                elif pre_com['type'] == 'fc':
                    in_channel = pre_com['size']
                else:
                    self.logger.error('build_weight() say Error component property, in conv else ')

                fc_size = com['size']

                if 'bias_transform' in com:
                    self.logger.info('in this com[bias_transform] = ' + str(com['bias_transform']))
                    name = com['bias_transform_name']
                    context = bias_variable([com['bias_transform']], name=name) 
                    in_channel += com['bias_transform']

                com['w'] = weight_variable([in_channel, fc_size],name= key + "_w") 
                com['b'] = bias_variable([fc_size], name = key + '_b') 

            pre_com = com

    def build_cnnlayer(self, conv_in, net_name):
        # build conv layer
        conv_out = None
        for key in sorted(cfg[net_name]):
            com = cfg[net_name][key]        #component
            if com['type'] in 'conv':
                stride = com['stride']
                conv_out = Conv2D(conv_in, com['kernel_size'], com['out_channel'], name_prefix=net_name+'_'+key, strides=[1, stride, stride, 1])
                conv_in  = conv_out
                if 'spatial_softmax' in com:
                    self.logger.debug('Last_conv.shape = {}'.format(conv_in.shape))
                    conv_out = tf.contrib.layers.spatial_softmax(conv_in, name='spatial_softmax')
        
        print(conv_out.shape)
        # if don't have spatial softmax need to flatten
        if len(conv_out.shape) > 2:
            conv_out = Flaten(conv_out)

        return conv_out

    def build_network(self, gif_pics):
		#-----------------Imitation Network-----------------#
        # define placeholder for inputs to network
        # x_image, ys= inp
        # print('inp = ' + str(inp))
        # print('before reshape -> gif_pics.shape = ' + str(gif_pics.shape))
        # print('before reshape -> ys.shape = ' + str(gif_actions.shape))
        # x_image = tf.placeholder(tf.float32, [None, 240, 240, 3], name='im_image') 
        # ys = tf.placeholder(tf.float32, [None, 2], name='im_pos')
        gif_len  = self.pics_each_gif if self.pics_each_gif  is not None else -1
        gif_pic, ext_pic = gif_pics

        gif_pics = tf.reshape(gif_pic, [self.batch_size, gif_len, self.img_h, self.img_w, self.img_d])
        ext_pics = tf.reshape(ext_pic, [self.batch_size, gif_len, self.ext_h, self.ext_w, self.img_d])

        # cnn layer
        gif_conv_out = self.build_cnnlayer(gif_pics, 'network')
        ext_conv_out = self.build_cnnlayer(ext_pics, 'network_ext')
        cnn_out = tf.concat([gif_conv_out, ext_conv_out], axis=1)

        # build fc layer
        name = 'im_fc_1'
        com = cfg['network_con'][name]
        fc_cnn_out = FC(cnn_out, com['size'], name_prefix=name, op=com['activation'])
        if self.drop_out:
            fc_cnn_out = tf.nn.dropout(fc_cnn_out, 0.5)

        # build feedback and bias layer
        name = 'im_fc_2'
        com = cfg['network_con'][name]
        context = tf.get_variable(com['bias_transform_name'])
        zero_tensor = tf.zeros_like(fc_cnn_out)[:, :com['bias_transform']]
        context = zero_tensor + context

        # self.logger.debug('zero_tensor {}'.format(zero_tensor))
        # self.logger.debug('after reshape context {}'.format(context.shape))
        # self.logger.debug('context {}'.format(context))

        fdb_out = tf.concat([self.batch_feedback, context], axis=1)
 
        fc_fdb_out = FC(fdb_out, com['size'], name_prefix=name, op=com['activation'])
        if self.drop_out:
            fc_fdb_out = tf.nn.dropout(fc_fdb_out, 0.5)

        fc_input = tf.concat([fc_cnn_out, fc_fdb_out], axis=1)
        print(fc_input.shape)

        name = 'im_fc_3'
        com = cfg['network_con'][name]
        fc_out = FC(fc_input, com['size'], name_prefix=name, op=com['activation'])
        if self.drop_out:
            fc_out = tf.nn.dropout(fc_out, 0.5)

        name = 'im_fc_4'
        com = cfg['network_con'][name]
        fc_out = FC(fc_out, com['size'], name_prefix=name, op=com['activation'])
        if self.drop_out:
            fc_out = tf.nn.dropout(fc_out, 0.5)

        name = 'im_prediction'
        com = cfg['network_con'][name]
        fc_out = FC(fc_out, com['size'], name_prefix=name, op=com['activation'])
        if self.drop_out:
            fc_out = tf.nn.dropout(fc_out, 0.5)
        
        # im_prediction
        return fc_out

    def build_prediction(self, in_elems):
        ''' in_elems include input_gif and label_action '''
        gif_pics, gif_acts = in_elems
        with tf.variable_scope("tower", reuse=tf.AUTO_REUSE): # tf.AUTO_REUSE #reuse=(True if i > 0 else None) ):
            gif_prediction = self.build_network(gif_pics)

            gif_dis  = euclidean(gif_prediction, gif_acts)
            gif_loss = tf.reduce_sum(gif_dis)

            # print('gif_loss.shape',gif_loss.shape)
        return [gif_prediction, gif_loss]

    def build_inputs_and_outputs(self, gif=None, ext=None, fdb=None, cmd=None): #, batch_gif_tensor):
        self.logger.debug('Start ---------- build_inputs_and_outputs() -----------')
        batch_gif_shape = [self.batch_size, self.pics_each_gif, self.img_h, self.img_w, self.img_d]
        batch_ext_shape = [self.batch_size, self.pics_each_gif, self.ext_h, self.ext_w, self.img_d]
        batch_act_shape = [self.batch_size, self.pics_each_gif, self.outs]
        batch_fdb_shape = [self.batch_size, self.pics_each_gif, self.feedback_num]

        self.batch_gif = tf.placeholder(tf.float32, batch_gif_shape, name='ph_batch_gif') if gif is None else gif
        self.batch_ext = tf.placeholder(tf.float32, batch_ext_shape, name='ph_batch_ext') if ext is None else ext
        self.batch_act = tf.placeholder(tf.float32, batch_act_shape, name='ph_batch_action')   if cmd is None else cmd
        self.batch_fdb = tf.placeholder(tf.float32, batch_fdb_shape, name='ph_batch_feedback') if fdb is None else fdb
        
        # fn = lambda x: self.build_prediction(x)
        # out_dtype = [tf.float32, tf.float32]
        # self.batch_result = tf.map_fn(fn, elems=(self.batch_gif, self.batch_action), dtype=out_dtype)
        self.batch_result = self.build_prediction([[self.batch_gif, self.batch_ext], self.batch_action])

        self.batch_prediction, self.batch_loss = self.batch_result
        self.total_im_loss = tf.reduce_mean(self.batch_loss)
        # self.total_im_loss = tf.reduce_sum(self.batch_loss)

        self.logger.debug('total_im_loss = {}'.format(self.total_im_loss))
        # self.batch_prediction, self.batch_loss
        # print('self.batch_prediction.shape',self.batch_prediction.shape)
        # print('self.batch_loss.shape',self.batch_loss.shape)
        # print('self.batch_result.shape',np.array(self.batch_result).shape)
        '''
        self.batch_prediction.shape (?, 20, 2)
        self.batch_loss.shape (?,)
        '''
        self.logger.debug('End ---------- build_inputs_and_outputs() -----------')
        
    def build_train_op(self):
        assert self.total_im_loss is not None, self.logger.error('build_train_op() say self.total_im_loss is None')      
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.total_im_loss)

    def set_network_property(self, drop_out=False):
        if drop_out: self.logger.info('Set Drop Out')
        self.drop_out = drop_out
