import pickle
import numpy as np
import os
import torch
import random
from tactile_generation_policy.dataset.base_dataset import BaseImageDataset
from tactile_generation_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from tactile_generation_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer
)
from tactile_generation_policy.common.action_utils import absolute_actions_to_relative_actions

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 root_dir,
                 delta_action=False,
                 relative_action=False,
                 mode='train'
                 ):
        self.root_dir = root_dir
        self.delta_action = delta_action
        self.relative_action = relative_action
        data_list = []
        # load data
        sample_list = sorted(os.listdir(root_dir))
        for i in sample_list:
            file_path = os.path.join(root_dir, i)
            sample = pickle.load(open(file_path, 'rb'))
            data_list.append(sample)
        
        random.shuffle(data_list)
        if mode == 'train':
            self.data_list = data_list[:int(0.9 * len(data_list))]
        else:
            self.data_list = data_list[int(0.9 * len(data_list)):]
        
    def __len__(self):
        return len(self.data_list)
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __getitem__(self, idx: int):
        sample_data = self.data_list[idx]
        image = sample_data['camera1_image']
        depth = sample_data['camera1_depth']
        state = sample_data['state']
        action = sample_data['action']
        tactile = sample_data['tactile'] # n, 700, 6

        if self.delta_action:
            new_action = np.concatenate((state, action), axis=0)
            action = np.diff(new_action)
        if self.relative_action:
            base_absolute_action = state
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)
        
        return {'observation': torch.tensor(tactile[0]),
                'action': torch.tensor(action),
                'future_img': torch.tensor(tactile[1:]),
                'condition_img': torch.tensor(image)}