from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
import random


class GraspDataset(Dataset):
    def __init__(self, train: bool=True) -> None:
        '''Dataset of successful grasps.  Each data point includes a 64x64
        top-down RGB image of the scene and a grasp pose specified by the gripper
        position in pixel space and rotation (either 0 deg or 90 deg)

        '''
        mode = 'train' if train else 'val'
        self.train = train
        data = np.load(f'{mode}_dataset.npz')
        self.imgs = data['imgs']
        self.actions = data['actions']

    def transform_grasp(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        '''Randomly rotate grasp by 0, 90, 180, or 270 degrees.

        Arguments
        ---------
        img:
            float tensor ranging from 0 to 1, shape=(3, 64, 64)
        action:
            array containing (px, py, rot_id), where px specifies the row in
            the image (heigh dimension), py specifies the column in the image (width dimension),
            and rot_id is an integer: 0 means 0deg gripper rotation, 1 means 90deg rotation.

        Returns
        -------
        tuple of (img, action) where both have been transformed by random
        rotation in the set (0 deg, 90 deg, 180 deg, 270 deg)

        Note
        ----
        The gripper is symmetric about 180 degree rotations so a 180deg rotation of
        the gripper is equivalent to a 0deg rotation and 270 deg is equivalent to 90 deg.

        Example Action Rotations
        ------------------------
        action = (32, 32, 1)
         - Rot   0 deg : rot_action = (32, 32, 1)
         - Rot  90 deg : rot_action = (32, 32, 0)
         - Rot 180 deg : rot_action = (32, 32, 1)
         - Rot 270 deg : rot_action = (32, 32, 0)

        action = (0, 63, 0)
         - Rot   0 deg : rot_action = ( 0, 63, 0)
         - Rot  90 deg : rot_action = ( 0,  0, 1)
         - Rot 180 deg : rot_action = (63,  0, 0)
         - Rot 270 deg : rot_action = (63, 63, 1)
        '''
        # action = [15,45,1]
        # print("action = ",action)
        [x_coordinate,y_coordinate,rotation_degree]= action
        img_size = img.shape[1]
        random_degree_rotation = random.randint(0, 3) 
        #at each iteration, it rotates the matrix by 90 by getting the transpose and reversing the matrix
        for i in range(1,random_degree_rotation+1):
            # getting the transpose and reversing the matrix
            row_reverse_coordinate = img_size - 1 - y_coordinate 
            y_coordinate = x_coordinate
            x_coordinate = row_reverse_coordinate


        if (random_degree_rotation%2) == 1:
            rotation_degree = (rotation_degree+1)%2
        if random_degree_rotation == 0:
            actual_rotation_degree = 0
        elif random_degree_rotation == 1:
            actual_rotation_degree = 90
        elif random_degree_rotation == 2:
            actual_rotation_degree = 180
        else:
            actual_rotation_degree = 270
        # print("random_degree_rotation = ",random_degree_rotation,"  actual_rotation_degree = ",actual_rotation_degree)
        rotated_image = T.functional.rotate(img,actual_rotation_degree) 
        action = [x_coordinate,y_coordinate,rotation_degree]
        # print("action sent back = ",action)
        # print("output action = ",action)

        return rotated_image, action

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = self.imgs[idx]
        action = self.actions[idx]

        H, W = img.shape[:2]
        img = TF.to_tensor(img)
        if np.random.rand() < 0.5:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        if self.train:
            img, action = self.transform_grasp(img, action)

        px, py, rot_id = action
        label = np.ravel_multi_index((rot_id, px, py), (2, H, W))

        return img, label

    def __len__(self) -> int:
        '''Number of grasps within dataset'''
        return self.imgs.shape[0]
