"""
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1

python3 gen_move-same_color.py --data_type=valid
"""
import numpy as np
import pyglet
from math import pi, cos, sin
import math
import random
import time
import imageio
from utils import recreate_dir, v_trapezoid, create_dir_not_exist, show_use_time
from argparse import ArgumentParser

pyglet.clock.set_fps_limit(10000)


class RegularPolygon:
    def __init__(self, N=5, r =30, x = 50, y = 50, theta = 0, color =[0,0,255]):
        self.N = N
        self.r = r
        self.set_x_y_theta_color(x, y, theta, color)

    def draw(self):
        self.poly_vert.draw(pyglet.gl.GL_POLYGON)

    def set_x_y_theta_color(self, x_center, y_center, theta, color ):
        vert = []
        for n in range(self.N):
            vert.append(self.r * math.cos( 2 * math.pi * n/self.N + theta) + x_center) # x 
            vert.append(self.r * math.sin( 2 * math.pi * n/self.N + theta) + y_center) # y

        # use float vertex and color 3 Byte 
        self.poly_vert =  pyglet.graphics.vertex_list(self.N,'v2f','c3B')
        self.poly_vert.vertices = vert
        self.poly_vert.colors = color * self.N


class Circle:
    # modify from https://gist.github.com/tsterker/1396796
    def __init__(self, x, y, r, color = [255, 0, 255]):
        self.r = r
        self.color = color
        self.set_x_y_r(x, y)

    def set_x_y_r(self, x, y, r = None):
        self.r = self.r if r == None else r
        iterations = int(2 * self.r * pi)
        s = sin(2*pi / iterations)
        c = cos(2*pi / iterations)

        dx, dy = self.r, 0

        vert = []
        for _ in range(iterations+1):
            dx, dy = (dx*c - dy*s), (dy*c + dx*s)
            vert.append(x+dx) # x 
            vert.append(y+dy) # x 

        # use float vertex and color 3 Byte 
        self.poly_vert =  pyglet.graphics.vertex_list(iterations+1,'v2f','c3B')
        self.poly_vert.vertices = vert
        self.poly_vert.colors = self.color *(iterations+1)

    def set_color(self,color = [255, 0, 255]):
        self.color = color
        iterations = int(2 * self.r * pi)
        self.poly_vert.colors = self.color *(iterations+1)

    def draw(self):
        self.poly_vert.draw(pyglet.gl.GL_TRIANGLE_FAN)


class Viewer(pyglet.window.Window):
    '''
        For render 
    '''
    color = {
        'background': [0]*3 + [0]
    }
    fps_display = pyglet.clock.ClockDisplay()
    # frame_count = 0
    output_dir = 'rawpic'

    def __init__(self, width, height, poly_r, mouse_in = None, predict_pos = None, caption = 'Polys', visible = True):
        super(Viewer, self).__init__(width, height, resizable=False, caption=caption, vsync=False)  # vsync=False to not use the monitor FPS
        
        self.width = width
        self.height = height
        
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])
    
        self.center_coord = np.array((min(width, height)/2, ) * 2)

        c1, c2, c3 = (255, 0, 0)*4, (255, 0, 0)*4, (0, 255, 0)*4
        self.label_1 = pyglet.text.Label(text="", x=10, y=height -20, font_size=8.0)
        self.label_2 = pyglet.text.Label(text="", x=10, y=height -40 , font_size=8.0)


        self.pos_list = []

        self.poly_triangle = RegularPolygon(N=3, r =poly_r, x = 120, y = 120, theta = 0 )
        self.poly_square =   RegularPolygon(N=4, r =poly_r, x = 60, y = 60, theta = 0 )
        self.poly_pentagon = RegularPolygon(N=5, r =poly_r, x = 30, y = 30, theta = 0 )
        
        # self.poly_triangle = RegularPolygon(N=6, r =poly_r, x = 120, y = 120, theta = 0 )
        # self.poly_square =   RegularPolygon(N=74, r =poly_r, x = 60, y = 60, theta = 0 )
        # self.poly_pentagon = RegularPolygon(N=4, r =poly_r, x = 30, y = 30, theta = 0 )

        self.circle = Circle(x = 0, y = 0, r = 5, color = [255, 255, 255])

        self.test_render = False

        # print('fps_display = ' + str(self.fps_display.get_fps()))

    def render(self, test_render = False, show_list = []):
        self.test_render = test_render
        self.show_list = show_list
        pyglet.clock.tick()

        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

        self.test_render = test_render
        

    def save_frame(self,f_path):
        pyglet.image.get_buffer_manager().get_color_buffer().save(f_path)

    def ret_frame_np(self):
        ''' return a fram which is numpy array '''
        img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        img_data = img.get_data('RGB', img.width * 3)
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        # img_np = img_np.reshape(240,240,3)
        img_np = img_np.reshape(self.height,self.width,3)
        img_np = img_np[::-1]

        # print('img_np.shape = ' + str(img_np.shape))
        return img_np

    def on_draw(self):
        self.clear()

        if self.show_list == None:
            self.show_list = []

        # print('self.show_list ', self.show_list )
        # print('in on_draw()')
        if 'tri' in self.show_list:
            self.poly_triangle.draw()
        if 'squ' in self.show_list:
            self.poly_square.draw()
        if 'pen' in self.show_list:
            self.poly_pentagon.draw()

        self.circle.draw()

        if self.test_render:
            
            self.label_1.draw()
            self.label_2.draw()

    
    def set_polys_property(self, in_dic):
        # self.polys_props = polys_props

        # print('in_dic ->', in_dic)
        p = in_dic['tri']
        self.poly_triangle.set_x_y_theta_color(p['x'], p['y'], p['theta'], p['color'])
        p = in_dic['squ']
        self.poly_square.set_x_y_theta_color(p['x'], p['y'], p['theta'], p['color'])
        p = in_dic['pen']
        self.poly_pentagon.set_x_y_theta_color(p['x'], p['y'], p['theta'], p['color'])

        #Matrix = [[0 for x in range(w)] for y in range(h)] 

    def set_circle_property(self, x, y, r = None):
        self.circle.set_x_y(x, y, r)

    def set_label_text(self, label_1_text = "", label_2_text = ""):
        self.label_1.text = label_1_text
        self.label_2.text = label_2_text

    @property
    def test_render(self):
        return self._test_render

    @test_render.setter
    def test_render(self, value):
        if not isinstance(value, bool):
            raise ValueError('test_render must be an bool!')
        self._test_render = value
    

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)


class MultiPolyEnv(object):
    viewer = None
    
    get_point = False
    mouse_in = np.array([False])

    viewer_xy = (100 , 100)

    poly_r = 10


    def __init__(self):
        self.target_rect_pos = np.array([self.viewer_xy[0] * 0.5, self.viewer_xy[1]* 0.5])
        self.center_coord = np.array(self.viewer_xy)/2
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, poly_r = self.poly_r, caption='Target')
        self.viewer2 = None

        self.viewer.render() # ignore first pic
        

    def get_random_pos(self):
        # x = np.clip(np.random.rand(1) * self.viewer_xy[0], 0 + self.poly_r, self.viewer_xy[0] - self.poly_r)
        # y = np.clip(np.random.rand(1) * self.viewer_xy[1], 0 + self.poly_r, self.viewer_xy[1] - self.poly_r)
        x = random.randint(self.poly_r, self.viewer_xy[0] - self.poly_r)
        y = random.randint(self.poly_r, self.viewer_xy[1] - self.poly_r)
        yaw = random.uniform(0, pi)
        return [x, y, yaw]

    

    def check_overlap(self, x,y, ary):
        
        def line_overlay_check(tar_x, x, check_r):
            # print('\t\t\tin line_overlay_check')
            tar_max_x =  tar_x + check_r
            tar_min_x =  tar_x - check_r
            max_x = x + check_r
            min_x = x - check_r
            if( (max_x > tar_min_x and  max_x < tar_max_x) \
             or (min_x > tar_min_x and  min_x < tar_max_x) ):
                return True
            return False

        # print('ary = ', ary)
        for p in ary: 
            t_or_f = line_overlay_check( p['x'], x, self.poly_r ) or \
                     line_overlay_check( p['y'], y, self.poly_r ) 
            if(t_or_f): 
                return True

        return False    # No overlay



    def gen_random_poly_pic(self,set_color_with_dic = None, viewer = None):

        viewer = self.viewer if viewer == None else viewer
        
        i = 0
        rand_poly_list = []
        while i < 3:
            # print('i = ', i)
            rand_pose = self.get_random_pos()
            if self.check_overlap(rand_pose[0], rand_pose[1], rand_poly_list):

                continue

            poly = {'x': rand_pose[0], 'y':rand_pose[1], 'theta': rand_pose[2], 'color': [random.randint(10,255) for _ in range(3)] }
            rand_poly_list.append(poly)

            i+=1


        # return  {'tri': rand_poly_list[0], 'squ':rand_poly_list[1], 'pen': rand_poly_list[2]}
        dic = {'tri': rand_poly_list[0], 'squ':rand_poly_list[1], 'pen': rand_poly_list[2]}
        
        if set_color_with_dic == None:
            choice = random.choice(['tri', 'squ', 'pen'])
            viewer.set_polys_property(dic)
        else: 
            choice = set_color_with_dic['choice']
            for k in set_color_with_dic:
                if k != 'choice':
                    dic[k]['color'] = set_color_with_dic[k]['color']

            viewer.set_polys_property(dic)

        dic['choice'] = choice

        return dic

    def main_ori(self, output_dir = 'train_data', num = 30):
        
        # self.output_dir = output_dir
        
        
        pos_list = []
        mimic_pos_list = []

        for frame_count in range(num):

            # frame_count += 1
            file_num=str(frame_count).zfill(5)

            # target imitation pic
            filename= output_dir + "/frame-"+file_num+'.png'
            im_dict = self.gen_random_poly_pic(f_path=filename, test_render=False)
            
            # target imitation position
            target_pos_poly =  im_dict[im_dict['choice'] ]
            pos_list.append([target_pos_poly['x'], target_pos_poly['y']])

            # mimic pic (mimic target)
            filename= output_dir + "/frame-"+file_num+'_mimic.png'
            mimic_dict = self.gen_random_poly_pic(f_path=filename, test_render=False, set_color_with_dic = im_dict)
            
            # mimic position
            mimic_pos_poly = mimic_dict[im_dict['choice']]
            mimic_pos_list.append([mimic_pos_poly['x'], mimic_pos_poly['y']])

        np.savetxt(output_dir + '/_target_pos.out', pos_list, fmt='%3d') 
        np.savetxt(output_dir + '/_mimic_pos.out', mimic_pos_list, fmt='%3d') 


    def only_render_one(self):
        
        tri = {'x': 100.0, 'y':100.0, 'theta': 0.2, 'color': [0, 0, 255]}
        squ = {'x': 50.0,  'y':50.0, 'theta': 0.15, 'color': [255, 0, 255]}
        pen = {'x': 30.0,  'y':30.0, 'theta': 0.1, 'color': [0, 255, 0]}
        dic = {'tri': tri, 'squ':squ, 'pen': pen}

        self.viewer.set_polys_property(dic)
        self.viewer.set_circle_property(20, 20)
        self.viewer.render(False)

        import time
        time.sleep(8)


   
    def v_trapezoid(t_now = 10,  t_all = 20, s = 150):
        '''
        speed use trapezoid
        (20 + 10 )* high_v / 2= 150 -> high_v =  150*2 / ( 20 + 10) 
        '''
        # half  timefor high speed
        l_r_len = t_all * 0.5 * 0.5   # trapezoid left & right len
        high_v = s * 2 / (t_all + t_all*0.5)

        if t_now < l_r_len:           # < 5
            return (t_now+1) * high_v / l_r_len
        elif t_now < (l_r_len + t_all * 0.5):
            return high_v
        elif t_now < (l_r_len + t_all * 0.5 + l_r_len):
            return (t_all - t_now -1) * high_v / l_r_len
        else:
            print('v_trapezoid() say Error t_now = ' + str(t_now))
            return 0

    def save_gif_test_render(self,move_target_pos, choice, choice_x, choice_y):
        
        txt_1 = "Target:  {} \t ({:5.2f},{:5.2f})".format(choice, move_target_pos[0], move_target_pos[1])
        txt_2 = "Now   :  {} \t ({:5.2f},{:5.2f})".format(choice, choice_x, choice_y)
        
        self.viewer.set_label_text(txt_1, txt_2)

    def save_gif(self, dic, gif_path = 'movie.gif', test_render = True):
        choice = dic['choice']

        # test_render = True
        cho_poly = dic[choice]
        img_list = []

        # how many picture in gif
        gif_num = 20


        # set move target and dt
        move_target_pos = self.get_random_pos()
        sub_pos = [move_target_pos[0]- cho_poly['x'] , move_target_pos[1]- cho_poly['y'] ]
        # move_dt = [sub_pos[0] / gif_num, sub_pos[1] / gif_num]

        self.viewer.circle.set_x_y_r(move_target_pos[0], move_target_pos[1], 5)
        
        v_list = []
        for i in range(gif_num):
            v = [v_trapezoid(i, gif_num, sub_pos[0]), v_trapezoid(i, gif_num, sub_pos[1])]
            cho_poly['x']  = cho_poly['x'] + v[0]
            cho_poly['y']  = cho_poly['y'] + v[1]
            v_list.append(v)
            if test_render:
                self.save_gif_test_render(move_target_pos, choice, cho_poly['x'], cho_poly['y'])

            self.viewer.set_polys_property(dic)
            
            # print('show_list', show_list)
            self.viewer.render(test_render,[choice] )
            img_np = self.viewer.ret_frame_np()
            img_list.append(img_np)
            # time.sleep(0.1)
            
        imageio.mimsave(gif_path, np.array(img_list))
        # print('v_list.shape =' + str(np.array(img_list).shape))
        np.savetxt(gif_path + '.v', np.array(v_list) ) 
        print('Save to ' + gif_path)

    def main(self, output_dir = 'test_out', load_poly_info =None, num = 1000):
        self.viewer.set_fps(200)
        recreate_dir(output_dir)
        circle_color = [random.randint(100,255) for _ in range(3)] if load_poly_info==None else load_poly_info.item().get('circle_color')
        # self.viewer.circle.set_color([random.randint(100,255) for _ in range(3)])
        self.viewer.circle.set_color(circle_color)
        
        # poly_info={'circle_color':circle_color, 'poly_type':dic['choice'], 'poly_color': dic[dic['choice']]['color']}
        # print(poly_info)
        # np.save("poly_info.npy", poly_info)

        
# print(load_poly_info)
# print(type(load_poly_info))
# print(load_poly_info.item().get('circle_color'))
# print(load_poly_info.item().get('poly_type'))
        dic = None
        if load_poly_info is not None:
            poly_type = load_poly_info.item().get('poly_type')
            poly_color = load_poly_info.item().get('poly_color')
            poly_tmp  = {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'color': poly_color }
            dic = {'choice':poly_type, poly_type: poly_tmp}  #dict()
        for i in range(num):
            dic = self.gen_random_poly_pic() if dic == None  else self.gen_random_poly_pic(dic)

            if i==0:
                poly_info={'circle_color':circle_color, 'poly_type':dic['choice'], 'poly_color': dic[dic['choice']]['color']}
                print(poly_info)
                np.save("poly_info.npy", poly_info)
            gif_path = '{}/{:0>4d}.gif'.format(output_dir, i) 
            self.save_gif(dic, gif_path,  test_render=False)

            
        # np.savetxt(output_dir + '/_mimic_pos.out', mimic_pos_list, fmt='%3d') 
        
    
        
    def get_train_color_dic(self):
        import os.path
        if os.path.isfile('poly_info.npy'):
            load_poly_info=np.load("poly_info.npy")
            # load_poly_info.item().get('poly_type')

            return load_poly_info
        else:
            return None

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data_type", dest='data_type', help="'train' or 'valid'", type=str, default='train')
    args = parser.parse_args()
    #---------Gen gif --------#
    env = MultiPolyEnv()

    load_poly_info = None
    if args.data_type == 'train':
    
        # env.set_fps(500)
        data_dir = 'train_data_same_color/'
        create_dir_not_exist(data_dir)
    else:
        data_dir = 'valid_data_same_color/'
        load_poly_info = env.get_train_color_dic()

    create_dir_not_exist(data_dir)
    s_time = time.time()
    env.main(data_dir,load_poly_info = load_poly_info, num=10000)
    show_use_time( time.time() - s_time, 'Use time:')
    # print('color time: {:6.2f}, all time: {:6.2f}'.format( time.time() - color_s_time, time.time() - s_time))

