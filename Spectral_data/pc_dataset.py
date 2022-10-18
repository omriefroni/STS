
import sys, torch, pickle
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import h5py
from itertools import combinations, product
from torch.utils.data import Subset
import random

from torchaudio import datasets



# STS static variable
SHREC_DATA_PATH = 'Spectral_data/datasets/shrec/shrec_all_120_lbo.hdf5'
SHREC_GT_PATH = 'Spectral_data/datasets/shrec/shrec_gt.hdf5'
SHREC_GEO = 'Spectral_data/datasets/shrec/shrec_geo_dist.hdf5'


SURREAL_TRAIN = 'Spectral_data/datasets//surreal/surreal_new_full_230k.hdf5'
SURREAL_TEST = 'Spectral_data/datasets/surreal/surreal_test.hdf5'


FAUST_TRAIN = 'Spectral_data/datasets//Faust_original/faust_train.hdf5' 
FAUST_TEST = 'Spectral_data/datasets/Faust_original/faust_test.hdf5'
FAUST_GEO = 'Spectral_data/datasets/Faust_original/faust_geo.hdf5' 


def create_sts_test_dataset(args):
    name =  args.STS_test_dataset
    if name == 'SHREC':
        dataset = SHREC(args,'test')
    elif name =='FAUST':
        dataset = FAUST(args,'test')
    return dataset

def create_sts_train_val_dataset(hparams, return_dataset=False):
    val_dataloader = None
    if hparams.STS_dataset=='SHREC':
        val_dataset = SHREC(hparams,'test') 
        train_dataset = SHREC(hparams,'train')
    elif hparams.STS_dataset=='SURREAL':
        val_dataset= SURREAL(hparams,'test')
        train_dataset = SURREAL(hparams,'train')

    elif hparams.STS_dataset=='FAUST':
        train_dataset = FAUST(hparams,'train')
        #TODO - add param for split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), len(train_dataset) - int(len(train_dataset)*0.9)])


        # another opption:
        # val_dataset = FAUST(hparams,'train')

    if return_dataset:
        return train_dataset, val_dataset
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers, drop_last=True)
    return train_dataloader, val_dataloader


class BasicPCDataset(Dataset):

    def __init__(self, hparams, dataset_type='train'):

        super(BasicPCDataset, self).__init__()
        self.dataset_type= dataset_type
        self.hparams = hparams
        

    def __len__(self):
        return len(self.all_pairs)

    # TODO - edit 
    def get_data_2_index(self, k1,k2):

        pair = k1 + '_' + k2
        out_dict = {'key': pair}
        keys_list =['verts', 'evects', 'a', 'evals'] 
        if self.dataset_type != 'train':
            keys_list = keys_list + ['geo_dist']

        data = self.data_h5

        if self.dataset_type in ['val','test'] and not self.gt_is_eys: # for SHREC dataset
            out_dict['gt'] = np.array(self.gt_h5[k1+'_'+k2])


        d1 = data[k1 + '_' + 'verts'] # source 
        d2 = data[k2 + '_' + 'verts'] # target


        min_num_points = min(d1.shape[0], d2.shape[0])
        do_sample=False
        if min_num_points > self.hparams.n_points:
            do_sample = True
            # Fixed indices for test and evaluation
            if self.dataset_type in ['val','test']: 
                if not self.gt_is_eys:
                    if 'indices' in self.gt_h5: # SHREC
                        indices = np.array(self.gt_h5['indices'])
                    elif 'indices' in data:  
                        indices = np.array(data['indices'])
                else:

                    if 'indices' in data: 
                        indices = np.array(data['indices'])
                    else:
                        indices = np.array(self.geo['indices'])
            else:
                indices = np.random.choice(min_num_points, self.hparams.n_points, replace=False)

            out_dict['indices'] = indices
            

            # Set the indices of the target ()
            if self.dataset_type in ['val','test'] and not self.gt_is_eys:
                out_dict['gt'] = out_dict['gt'][indices]
                indices_2 = out_dict['gt'].astype(int)
            else:
                indices_2 = indices
            

        for kk in keys_list:
            if kk == 'geo_dist':
                key = k1 + '_' + kk
                # TODO - fix me
                if key in self.geo.keys():
                    d1 = np.array(self.geo[k1 + '_' + kk])
                    d2 = np.array(self.geo[k2 + '_' + kk]) 
                else:
                    d1 = np.array(self.geo[k1 + '_' + kk.replace('geo','goe')])
                    d2 = np.array(self.geo[k2 + '_' + kk.replace('geo','goe')]) 
            else:
                d1 = np.array(data[k1 + '_' + kk])
                d2 = np.array(data[k2 + '_' + kk])
           
            if kk =='a':
                if len(d1.shape) == 2:
                    d1 = np.diag(d1)
                    d2 = np.diag(d2)
            
            if do_sample and kk in ['verts', 'evects','a', 'geo_dist', 'faces' ]:
                d1 = d1[indices]
                d2 = d2[indices_2]
                if kk == 'geo_dist':
                    d1 = d1[:,indices]
                    d2 = d2[:,indices_2]

            out_dict[kk]  = np.concatenate([d1[...,None], d2[...,None]], axis=-1)


        if self.dataset_type == 'train':
            out_dict['geo_dist'] = []
        
        out_dict['evects'] = out_dict['evects'][:,:self.hparams.k_lbo,:]
        out_dict['evals'] = out_dict['evals'][:self.hparams.k_lbo,:]

        return out_dict


class SHREC(BasicPCDataset):

    def __init__(self, hparams, dataset_type='train'):

        super(SHREC, self).__init__(hparams, dataset_type)

        self.data_h5 = h5py.File(SHREC_DATA_PATH, 'r')
        self.geo = h5py.File(SHREC_GEO, 'r')
        if dataset_type in ['val','test']:
            self.gt_h5 = h5py.File(SHREC_GT_PATH, 'r')
        self.gt_is_eys = False
        self.get_pair()


    def __getitem__(self, index):
        pair = self.all_pairs[index]
        k1 = str(pair[0])
        k2 = str(pair[1])
        return self.get_data_2_index(k1,k2)


    def get_pair(self):
        if self.dataset_type == 'train':
            num_of_shapes = [int(k.split("_")[0]) for k in self.data_h5.keys() if '_evects' in k]
            self.all_pairs= list(combinations(num_of_shapes,2))
        else:
            all_gt_keys = self.gt_h5.keys()
            self.all_pairs = [(int(key.split('_')[0]), int(key.split('_')[1])) for key in all_gt_keys if key != 'indices']


class SURREAL(BasicPCDataset):

    def __init__(self, hparams, dataset_type='train'):

        super(SURREAL, self).__init__(hparams, dataset_type)
        if dataset_type == 'train':
            self.data_h5 = h5py.File(SURREAL_TRAIN, 'r')
        elif dataset_type == 'test':
            self.data_h5 = h5py.File(SURREAL_TEST, 'r')
            self.geo = self.data_h5
        self.dataset_type = dataset_type

        self.gt_is_eys = True

        self.get_pair()

    def get_pair(self):

        if self.dataset_type == 'train':
            list_path = SURREAL_TRAIN.replace('.hdf5', '_shapes.pkl')
            with open(list_path, 'rb') as ff:
                num_of_shapes = pickle.load(ff)

            # num_of_shapes = [int(k.split("_")[0]) for k in self.data_h5.keys() if '_evects' in k]
            self.L = len(num_of_shapes) //2

            self.all_shpaes = num_of_shapes
        else:
            self.all_pairs= list(self.data_h5['test_samples'][:])
            self.L = len(self.all_pairs)

    
    def __len__(self):
        return self.L 
        
    def __getitem__(self, index):
        if self.dataset_type =='train':
            random.seed(index) 
            t1 = [self.all_shpaes.pop(random.randrange(len(self.all_shpaes))) for _ in range(2)]
            assert(t1[0]!=t1[1])
            k1 = str(t1[0])
            k2 = str(t1[1])
        else:
            pair = self.all_pairs[index]
            k1 = str(pair[0])
            k2 = str(pair[1])
        return self.get_data_2_index(k1,k2)



class FAUST(BasicPCDataset):

    def __init__(self, hparams, dataset_type='train'):

        super(FAUST, self).__init__(hparams, dataset_type)
        if dataset_type == 'train':
            self.data_h5 = h5py.File(FAUST_TRAIN, 'r')
            self.geo = h5py.File(FAUST_GEO, 'r')
        else:
            self.data_h5 = h5py.File(FAUST_TEST, 'r')
            self.geo = h5py.File(FAUST_GEO, 'r')

        self.dataset_type = dataset_type
        self.gt_is_eys = True
        self.get_pair()

    def get_pair(self):

        self.shape_list = [k.replace('_evects','') for k in self.data_h5.keys() if '_evects' in k ]
        self.all_pairs= list(combinations(list(range(len(self.shape_list ))),2))

    def __len__(self):
        return len(self.all_pairs)
        
    def __getitem__(self, index):
        pair_ind = self.all_pairs[index]
        k1=str(self.shape_list[pair_ind[0]])
        k2=str(self.shape_list[pair_ind[1]])
        return self.get_data_2_index(k1,k2)




# if __name__ == '__main__':


