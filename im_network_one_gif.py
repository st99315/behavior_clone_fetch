import tensorflow as tf
from DNN_v2 import *
from config import cfg
import math


def euclidean(x,y, multi=100.0):
    """
      x:      [0,0],[1,1],[2,2]
      y:      [3,4 ],[7,9],[5,5]
      output: [  5.         10.          4.2426405]
    """
    multiplier = tf.constant(multi, dtype='float') #for bc #10000
    sub =x*multiplier-y*multiplier
    
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
    def __init__(self, training=True):
        self.batch_size = cfg['batch_size']
        self.pic_num_each_gif = cfg['pic_num_each_gif']
        self.img_w = cfg['image_width']
        self.img_h = cfg['image_height']
        self.img_d = cfg['image_depth']
        self.feedback_num = cfg['robot_configuration_num']
        self.outs = cfg['network']['im_prediction']['size']
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
        sort_net_key = sorted(cfg['network'])  
        pre_com = None
        fc_down_factor = 0
        for key in sort_net_key:
            com = cfg['network'][key]        #component
            
            if com['type'] == 'conv':
                if pre_com == None:
                    in_channel = self.img_d
                elif pre_com['type'] == 'conv':
                    in_channel = pre_com['out_channel']
                else:
                    print('build_weight() say Error component property, in conv else ')
                w_shape = [com['kernel_size'], com['kernel_size'], in_channel, com['out_channel']]
                com['w'] = weight_variable(w_shape , name= key + "_w") 
                com['b'] = bias_variable([com['out_channel']]  , name= key + "_b")
                fc_down_factor = fc_down_factor * int(com['stride']) if fc_down_factor is not 0 else  int(com['stride'])
            elif  com['type'] == 'fc':
                if pre_com['type'] == 'conv':
                    in_channel = self.feedback_num + math.ceil(self.img_w/ fc_down_factor) * math.ceil(self.img_h/ fc_down_factor) * pre_com['out_channel'] #because default padding is 'SAME'
                    print('first layer')
                    print('\t fc_down_factor = '+ str(fc_down_factor))
                    print('\t in_channel = '+ str(in_channel))

                elif pre_com['type'] == 'fc':
                    in_channel =  pre_com['size']
                else:
                    print('build_weight() say Error component property, in conv else ')
                fc_size = com['size']

                if 'bias_transform' in com:
                    print('in this com[bias_transform] = ' + str(com['bias_transform']))
                    n =  com['bias_transform_name']
                    context = bias_variable([com['bias_transform'] ], name = n) 
                    in_channel += com['bias_transform']

                com['w'] = weight_variable([in_channel, fc_size],name= key + "_w") 
                com['b'] = bias_variable([fc_size], name = key + '_b') 

            pre_com = com
  
    def build_network(self, gif_pics):
		#-----------------Imitation Network-----------------#
        # define placeholder for inputs to network
        # x_image, ys= inp
        # print('inp = ' + str(inp))
        # print('before reshape -> gif_pics.shape = ' + str(gif_pics.shape))
        # print('before reshape -> ys.shape = ' + str(gif_actions.shape))
        # x_image = tf.placeholder(tf.float32, [None, 240, 240, 3], name='im_image') 
        # ys = tf.placeholder(tf.float32, [None, 2], name='im_pos')
        gif_len = self.pic_num_each_gif if self.pic_num_each_gif is not None else -1
        gif_pics = tf.reshape(gif_pics, [gif_len, self.img_h, self.img_w, self.img_d])
        # gif_pics = tf.reshape(gif_pics, [-1, self.img_w, self.img_h, self.img_d])

        # gif_actions =  tf.reshape(gif_actions, [-1, 2])
        # print('after reshape -> gif_pics.shape = ' + str(gif_pics.shape))

        conv1 = Conv2D(gif_pics, 3, 30, name_prefix='im_conv_1')
        conv2 = Conv2D(conv1, 3, 30, name_prefix='im_conv_2')
        conv3 = Conv2D(conv2, 3, 30, name_prefix='im_conv_3')

        # conv1 = Conv2D(gif_pics, 3, 32, name_prefix='im_conv_1')
        # conv2 = Conv2D(conv1, 3, 48, name_prefix='im_conv_2')
        # conv3 = Conv2D(conv2, 3, 64, name_prefix='im_conv_3')
        # conv4 = Conv2D(conv3, 3, 128, name_prefix='im_conv_4')

        flat  = Flaten(conv3)
        fc_input = tf.concat([flat, self.batch_feedback], axis=1)

        print('before contact, im_flat', fc_input)
        context = tf.get_variable('context')
        print('before reshape context', context)

        zero_tensor = tf.zeros_like(fc_input)[:, :10]
        context = zero_tensor + context
        fc_input = tf.concat([fc_input, context], axis=1)

        print('zero_tensor', zero_tensor)
        print('after reshape context', context)
        print('context', context)
        print('im_flat', fc_input)

        fc1 = FC(fc_input, 200, name_prefix = 'im_fc_1', op='none')
        if self.drop_out: 
            print('Use drop_out!')
            fc1 = tf.nn.dropout(fc1, 0.5)
        
        fc2 = FC(fc1, 200, name_prefix = 'im_fc_2', op='none')
        if self.drop_out: fc2 = tf.nn.dropout(fc2, 0.5)

        im_prediction = FC(fc2, self.outs, name_prefix='im_prediction', op='none')
        
        # print('flat ->', flat)
        # print('fc1 -> ', fc1)
        # print('fc2 -> ', fc2)
        # print('im_prediction -> ', im_prediction)
        
        # im_dis = euclidean(im_prediction, gif_actions)
        # im_loss = tf.reduce_mean(im_dis)
        
        # fn_output = [im_prediction, im_loss]
        # return [im_prediction, im_loss]
        return im_prediction

    def build_prediction(self, in_elems):
        gif_pics, gif_actions = in_elems
        with tf.variable_scope("tower", reuse=True): # tf.AUTO_REUSE #reuse=(True if i > 0 else None) ):
            gif_prediction = self.build_network(gif_pics)
            # print('gif_prediction.shape = ' + str(gif_prediction.shape))
            # print('gif_actions.shape = ' + str(gif_actions.shape))
            gif_dis = euclidean(gif_prediction, gif_actions)
            # for i in range(1000):
                # print('gif_dis.shape = ' + str(gif_dis.shape))
            # shape: (20, 1)
            # gif_loss = tf.reduce_mean(gif_dis)
            gif_loss = tf.reduce_sum(gif_dis)

            # print('gif_prediction.shape',gif_prediction.shape)
            # print('gif_loss.shape',gif_loss.shape)
        return [gif_prediction, gif_loss]

    def build_inputs_and_outputs(self, gif=None, fdb=None, cmd=None): #, batch_gif_tensor):

        print('START----------build_inputs_and_outputs()-----------')
        batch_gif_shape = [self.pic_num_each_gif, self.img_h, self.img_w, self.img_d]
        # batch_gif_shape = [None self.img_h, self.img_w, self.img_d]

        self.batch_gif = tf.placeholder(tf.float32, batch_gif_shape, name='batch_gif') if gif is None else gif
        self.batch_action   = tf.placeholder(tf.float32, [self.pic_num_each_gif, self.outs], name='batch_gif_action') if cmd is None else cmd
        self.batch_feedback = tf.placeholder(tf.float32, [self.pic_num_each_gif, self.feedback_num], name='batch_gif_feedback') if fdb is None else fdb

        #self.batch_action = tf.placeholder(tf.float32, [None, 11], name='batch_gif_action') 
        
        # fn = lambda x: self.build_prediction(x)
        # out_dtype = [tf.float32, tf.float32]
        # self.batch_result = tf.map_fn(fn, elems=(self.batch_gif, self.batch_action), dtype=out_dtype)
        self.batch_result = self.build_prediction([self.batch_gif, self.batch_action])

        self.batch_prediction, self.batch_loss = self.batch_result
        self.total_im_loss = tf.reduce_mean(self.batch_loss)
        # self.total_im_loss = tf.reduce_sum(self.batch_loss)
        print('total_im_loss= ', self.total_im_loss)
        # # self.batch_prediction, self.batch_loss
        # print('self.batch_prediction.shape',self.batch_prediction.shape)
        # print('self.batch_loss.shape',self.batch_loss.shape)
        # print('self.batch_result.shape',np.array(self.batch_result).shape)
        '''
        self.batch_prediction.shape (?, 20, 2)
        self.batch_loss.shape (?,)
        '''
        print('END----------build_inputs_and_outputs()-----------')
        
    def build_train_op(self):
        assert self.total_im_loss is not None, 'build_train_op() say self.total_im_loss is None'
        
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.total_im_loss)
        # return batch_result

    def set_network_property(self, drop_out=False):
        self.drop_out = drop_out

    def make_gif_batch(self, all_filenames): #, train=True):
        ''' Modify from https://github.com/tianheyu927/mil '''
        # TEST_INTERVAL = 500
        batch_image_size = self.batch_size  # TODO: be im & mimic

        # if train:
        #     all_filenames = self.all_training_filenames
        #     if restore_iter > 0:
        #         all_filenames = all_filenames[batch_image_size*(restore_iter+1):]
        # else:
        #     all_filenames = self.all_val_filenames
        #     if restore_iter > 0:
        #         all_filenames = all_filenames[batch_image_size*(int(restore_iter/TEST_INTERVAL)+1):]
        print('START----------make_batch_tensor-----------')
        # print('make_batch_tensor say: all_filenames ={}')
        
        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        # print 'Generating image processing ops'
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_gif(image_file)
        # should be T x C x W x H
        image.set_shape((self.pic_num_each_gif, self.img_h, self.img_w, self.img_d))
        image = tf.cast(image, tf.float32)
        image /= 255.0
        # image = tf.transpose(image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        image = tf.reshape(image, [self.pic_num_each_gif, -1])
        min_queue_examples = 64 #128 #256
        print('[I] Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,  # 2(update_batch+test_batch) * 5 (meta)
                num_threads= 1,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_images = []

        for i in range(self.batch_size):
            image = images[i:(i+1)]
            # from before make_batch_tensor say: update_batch_size=1, test_batch_size=1
            # print('\t image before reshape = ' , image.shape)
            image = tf.reshape(image, [1*self.pic_num_each_gif, -1])
            # print('\t image after reshape = ' , image.shape)
            all_images.append(image)
        print('all_images', all_images)
        print('END----------make_batch_tensor-----------')
        return tf.stack(all_images, name='make_gif_batch')
