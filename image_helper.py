import logging
logger = logging.getLogger('logger')

import torch
import torch.utils.data

from helper import Helper
from torchvision import  transforms
from utils.celeba_dataset import CelebADataset


# Potentially not needed
# from collections import defaultdict, OrderedDict
#from torchvision import datasets
# import random
# import torchvision
# import os


#Old TODO delete
#import numpy as np
#from models.simple import SimpleNet #TODO check if needed


POISONED_PARTICIPANT_POS = 0


class ImageHelper(Helper):

    def poison(self):
        return

    
    def load_celeba_data(self):
    """Build and return a data loader."""
    self.name = self.params['name']
    
    #TODO maybe not even needed? since instant use in the files?
    #TODO wouldn't it be cleaner to initiallize it at the beginning and then just pass it rather then lookup?
    image_dir = ''
    attr_path = ''
    selected_attrs = ''
    
    crop_size = 178 #178 #TODO maybe lower image resolution?
    image_size = 128 #128 #TODO pixels find me
    
    flip = transforms.RandomHorizontalFlip()
    crop = transforms.CenterCrop(crop_size)
    resize = transforms.Resize(image_size)
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    transform_train = transforms.Compose([flip, crop, resize, transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([crop, resize, transforms.ToTensor(), normalize])

    self.train_dataset = CelebADataset(image_dir=self.params['image_dir'],
                                        attr_path=self.params['attr_path'],
                                        selected_attr=self.params['selected_attr'],
                                        protected_attr=self.params['protected_attr'],
                                        mode='train',
                                        transform=transform_train)

    self.test_dataset = CelebADataset(image_dir=self.params['image_dir'],
                                        attr_path=self.params['attr_path'],
                                        selected_attr=self.params['selected_attr'],
                                        protected_attr=self.params['protected_attr'],
                                        mode='test',
                                        transform=transform_test)
    
    self.dataset_size = len(self.train_dataset)
    logger.info(f"Length of CelebA dataset: {self.dataset_size}")
    
    #Maybe the different shuffles make the total a different mix?
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.params['batch_size'],
                                                    shuffle=True,
                                                    num_workers=2,
                                                    drop_last=True)

    self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.params['test_batch_size'],
                                                    shuffle=False, #TODO changed this to true for although what is the point...
                                                    num_workers=2)
            
    self.labels = self.params['labels']
    return True





# #TODO check what does and if needed
#     def sampler_per_class(self):
#         self.per_class_loader = OrderedDict()
#         per_class_list = defaultdict(list)
#         for ind, x in enumerate(self.test_dataset):
#             _, label = x
#             per_class_list[int(label)].append(ind)
#         per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
#         for key, indices in per_class_list.items():
#             self.per_class_loader[int(key)] = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params[
#                 'test_batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))




# #TODO check what these sampelers do and if we need them.?
#     def sampler_exponential_class(self, mu=1, total_number=40000, key_to_drop=False, number_of_entries=False):
#         per_class_list = defaultdict(list)
#         sum = 0
#         for ind, x in enumerate(self.train_dataset):
#             _, label = x
#             sum += 1
#             per_class_list[int(label)].append(ind)
#         per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
#         unbalanced_sum = 0
#         for key, indices in per_class_list.items():
#             if key and key != key_to_drop:
#                 unbalanced_sum += len(indices)
#             elif key and key == key_to_drop:
#                 unbalanced_sum += number_of_entries
#             else:
#                 unbalanced_sum += int(len(indices) * (mu ** key))

#         if key_to_drop:
#             proportion = 1
#         else:
#             if total_number / unbalanced_sum > 1:
#                 raise ValueError(
#                     f"Expected at least {total_number} elements, after sampling left only: {unbalanced_sum}.")
#             proportion = total_number / unbalanced_sum
#         logger.info(sum)
#         ds_indices = list()
#         subset_lengths = list()
#         sum = 0
#         for key, indices in per_class_list.items():
#             random.shuffle(indices)
#             if key and key != key_to_drop:
#                 subset_len = len(indices)
#             elif key and key == key_to_drop:
#                 subset_len = number_of_entries
#             else:
#                 subset_len = int(len(indices) * (mu ** key) * proportion)
#             sum += subset_len
#             subset_lengths.append(subset_len)
#             logger.info(f'Key: {key}, len: {subset_len} class_len {len(indices)}')
#             ds_indices.extend(indices[:subset_len])
#         logger.info(sum)
#         self.dataset_size = sum
#         logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
#         self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params[
#             'batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices), drop_last=True)

#     def sampler_exponential_class_test(self, mu=1, key_to_drop=False, number_of_entries_test=False):
#         per_class_list = defaultdict(list)
#         sum = 0
#         for ind, x in enumerate(self.test_dataset):
#             _, label = x
#             sum += 1
#             per_class_list[int(label)].append(ind)
#         per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
#         unbalanced_sum = 0
#         for key, indices in per_class_list.items():
#             unbalanced_sum += int(len(indices) * (mu ** key))

#         logger.info(sum)
#         ds_indices = list()
#         subset_lengths = list()
#         sum = 0
#         for key, indices in per_class_list.items():
#             random.shuffle(indices)
#             if key and key != key_to_drop:
#                 subset_len = len(indices)
#             elif key and key == key_to_drop:
#                 subset_len = number_of_entries_test
#             else:
#                 subset_len = int(len(indices) * (mu ** key))
#             sum += subset_len
#             subset_lengths.append(subset_len)
#             logger.info(f'Key: {key}, len: {subset_len} class_len {len(indices)}')
#             ds_indices.extend(indices[:subset_len])
#         logger.info(sum)
#         logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
#         self.test_loader_unbalanced = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params[
#             'batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices), drop_last=True)

# #TODO check if loader needed
#     def create_loaders(self):
#         self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
#                                                         batch_size=self.params['batch_size'],
#                                                         shuffle=True, drop_last=True)
#         self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
#                                                        batch_size=self.params['test_batch_size'],
#                                                        shuffle=True)



#TODO needed might be helpfull in future?
    # def balance_loaders(self):
    #     per_class_index = defaultdict(list)
    #     for i in range(len(self.train_dataset)):
    #         _, label = self.train_dataset.samples[i]
    #         per_class_index[label].append(i)
    #     total_indices = list()
    #     if self.params['inat_drop_proportional']:
    #         for key, value in per_class_index.items():
    #             random.shuffle(value)
    #             per_class_no = int(len(value) * (self.params['ds_size'] / len(self.train_dataset)))
    #             logger.info(f'class: {key}, len: {len(value)}. new length: {per_class_no}')
    #             total_indices.extend(value[:per_class_no])
    #     else:
    #         per_class_no = self.params['ds_size'] / len(per_class_index)
    #         for key, value in per_class_index.items():
    #             logger.info(f'class: {key}, len: {len(value)}. new length: {per_class_no}')
    #             random.shuffle(value)
    #             total_indices.extend(value[:per_class_no])
    #     logger.info(f'total length: {len(total_indices)}')
    #     self.dataset_size = len(total_indices)
    #     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=total_indices)
    #     self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
    #                                                     batch_size=self.params['batch_size'],
    #                                                     sampler=train_sampler,
    #                                                     num_workers=2, drop_last=True)


#TODO Might be usefull but will have to use it 
    # def get_unbalanced_faces(self):
    #     self.unbalanced_loaders = dict()
    #     files = os.listdir(self.params['folder_per_class'])
    #     # logger.info(files)
    #     for x in sorted(files):
    #         indices = torch.load(f"{self.params['folder_per_class']}/{x}")
    #         # logger.info(f'unbalanced: {x}, {len(indices)}')
    #         sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
    #         self.unbalanced_loaders[x] = torch.utils.data.DataLoader(self.test_dataset,
    #                                                     batch_size=self.params['test_batch_size'],
    #                                                     sampler=sampler,
    #                                                     num_workers=2, drop_last=True)
    #     return True


    

    
    
    # def load_jigsaw(self):
    #     import pickle
    #     import pandas as pd

    #     max_features = 50000

    #     train = pd.read_csv('data/jigsaw/processed_train.csv')
    #     test = pd.read_csv('data/jigsaw/processed_test.csv')
    #     # after processing some of the texts are emply
    #     train['comment_text'] = train['comment_text'].fillna('')
    #     test['comment_text'] = test['comment_text'].fillna('')
    #     with open(f'data/jigsaw/tokenizer_{max_features}.pickle', 'rb') as f:
    #         tokenizer = pickle.load(f)

    #     X_train = tokenizer.texts_to_sequences(train['comment_text'])
    #     X_test = tokenizer.texts_to_sequences(test['comment_text'])
    #     x_train_lens = [len(i) for i in X_train]
    #     x_test_lens = [len(i) for i in X_test]

    # def create_model(self):
    #     return

    # def plot_acc_list(self, acc_dict, epoch, name, accuracy):
    #     import matplotlib
    #     matplotlib.use('AGG')
    #     import matplotlib.pyplot as plt

    #     acc_list = sorted(acc_dict.items(), key=lambda t: t[1])
    #     sub_lists = list()
    #     names = list()
    #     for x, y in acc_list:
    #         sub_lists.append(y)
    #         names.append(str(x))
    #     fig, ax = plt.subplots(1, figsize=(40, 10))
    #     ax.plot(names, sub_lists)
    #     ax.set_ylim(0, 100)
    #     ax.set_xlabel('Labels')
    #     ax.set_ylabel('Accuracy')
    #     fig.autofmt_xdate()
    #     plt.title(f'Accuracy plots. Epoch {epoch}. Main accuracy: {accuracy}')
    #     plt.savefig(f'{self.folder_path}/figure__{name}_{epoch}.pdf', format='pdf')

    #     return fig
