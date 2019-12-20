import collections
import copy
import json
import os
import time
import gym
from gym.envs.registration import register
import gym.spaces
import networkx as nx 
import numpy as np 
import scipy.io as sio 


import cv2
import random
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#import visualization_utils as vis_util 
import pickle

TRAIN_WORLDS = [
    'Home_014_1'
]

register(id='active-vision-env-v0',entry_point='cognitive_planning.envs.active_vision_dataset_env:ActiveVisionDatasetEnv',)

SUPPORTED_ACTIONS=['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']
_Graph=collections.namedtuple('_Graph',['graph','id_to_index','index_to_id'])

with open ('./jsonfile/imagelist', 'rb') as ft:
    imagelist=pickle.load(ft)

def _get_image_folder(root,world):
    return os.path.join(root, world,'jpg_rgb')

def _get_json_path(root, world):
    return os.path.join(root, world, 'annotations.json')

def _get_image_path(root, world, image_id):
    return  os.path.join(_get_image_folder(root, world),image_id+'.jpg')

def _get_image_list(path, world):
    """builds a dictionary for all the worlds.
    Args:
    path: the path to the dataset on cns.
    worlds: list of the worlds.
    returns:
    dictionary where the key is the world names and
    the values are the image_ids of that world
    """
    world_id_dict={}
    
    files=[t[:-4] for t in os.listdir(_get_image_folder(path, world))]
    world_id_dict[world]=files
    return world_id_dict

def read_all_poses(dataset_root, world):
    """reads all the poses for each world
    Args:
    dataset_root: the path to the root of the dataset.
    world: string, name of the world
    Returns:
    dictonary of poses for all the images in each world. The key is the image id of each view
    and the values are tuple of (x,z,R, scale).
    where x and z are the first and third coordinate of translation. R is the 3X3 rotation matrix
    and scale is a float scalar that indicates the scale that needs to be multipled to x and z in order to get the real world coordicates.
    """
    path = os.path.join(dataset_root, world, 'image_structs.mat')
    with tf.gfile.Open(path) as f:
        data = sio.loadmat(f)
    xyz = data['image_structs']['world_pos']
    image_names = data['image_structs']['image_name'][0]
    rot = data['image_structs']['R'][0]
    scale = data['scale'][0][0]
    n = xyz.shape[1]
    x = [xyz[0][i][0][0] for i in range(n)]
    z = [xyz[0][i][2][0] for i in range(n)]
    names = [name[0][:-4] for name in image_names]
    if len(names) != len(x):
        raise ValueError('number of image names are not equal to the number of '
                        'poses {} != {}'.format(len(names), len(x)))
    output = {}
    for i in range(n):
        if rot[i].shape[0] != 0:
            assert rot[i].shape[0] == 3
            assert rot[i].shape[1] == 3
            output[names[i]] = (x[i], z[i], rot[i], scale)
        else:
            output[names[i]] = (x[i], z[i], None, scale)

    return output
    
ACTIONS=['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']


class ActiveVisionDatasetEnv():
    """simulates the environment from ActiveVisionDataset."""
    cached_data=None
    def __init__(self,world,goal_image_id=None, dataset_root='./jsonfile/', actions=ACTIONS):
        self._dataset_root=dataset_root
        self._actions=ACTIONS
        self._cur_world=world
        self._world_id_dict={}
        self._world_id_dict[self._cur_world]=imagelist[self._cur_world]

        self._all_graph = {}
        with open(_get_json_path(self._dataset_root, self._cur_world), 'r') as f:
            file_content = f.read()
            file_content = file_content.replace('.jpg', '')
            io = StringIO(file_content)
            self._all_graph[self._cur_world] = json.load(io)
        self.graph=nx.DiGraph()
        self.id_to_index={}
        self.index_to_id={}
        self.image_image_action={}
        image_list=self._world_id_dict[self._cur_world]
        imlist=[]
        for image_id in image_list[self._cur_world]:
            self.image_image_action[image_id]={} 
            for action in self._actions:
                if action=='stop':
                    self.image_image_action[image_id][image_id]=action
                    continue
                next_image=self._all_graph[self._cur_world][image_id][action]
                if next_image:
                    self.image_image_action[image_id][next_image]=action
  

        for i, image_id in enumerate(image_list[self._cur_world]):
            self.id_to_index[image_id]=i
            self.index_to_id[i]=image_id
            self.graph.add_node(i)
        for image_id in image_list[self._cur_world]:
            for action in self._actions:
                if action=='stop':
                    continue
                next_image=self._all_graph[self._cur_world][image_id][action]

                if next_image:
                    self.graph.add_edge(self.id_to_index[image_id],self.id_to_index[next_image],action=action)
        self.n_locations=self.graph.number_of_nodes()
        if goal_image_id:
            self.goal_image_id=goal_image_id
            self.goal_vertex=self.id_to_index[self.goal_image_id]
        else:
            self.goal_vertex=random.randrange(self.n_locations)
            self.goal_image_id=self.index_to_id[self.goal_vertex]
        self.reset()

        
    def reset(self):
        while True:
            k = random.randrange(self.n_locations)
            min_d = np.inf
            path = nx.shortest_path(self.graph,k,self.goal_vertex)
            min_d=min(min_d,len(path))
            if min_d>0:
                break
        self.current_vertex=k
        self._cur_image_id=self.index_to_id[self.current_vertex]
        self._steps_taken=0
        self.reward   = 0
        self.collided = False
        self.done = False

    def step(self, action):
        action = self._actions[action]
        next_image_id = self._all_graph[self._cur_world][self._cur_image_id][action]            
        self._cur_image_id = next_image_id
        self.current_vertex=self.id_to_index[self._cur_image_id]

    def shortest_path(self,vertex,goal):
        path=nx.shortest_path(self.graph, vertex, goal)
        return path
    def all_shortest_path(self,start, end):
        path=nx.all_shortest_paths(self.graph, start, end)
        return path

    

if __name__ == "__main__":
    env = ActiveVisionDatasetEnv(world="Home_001_1")
    print("end")



    







   



        












        




























