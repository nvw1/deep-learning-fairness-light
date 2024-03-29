# Based on https://github.com/FarrandTom/deep-learning-fairness
# from @article{farrand2020neither, title={Neither Private Nor Fair: Impact of Data Imbalance on Utility and Fairness in Differential Privacy}, author={Farrand, Tom and Mireshghallah, Fatemehsadat and Singh, Sahib and Trask, Andrew}, journal={arXiv preprint arXiv:2009.06389}, year={2020} }

import logging
logger = logging.getLogger('logger')

import torch.utils.data


from helper import Helper
from torchvision import  transforms
from utils.celeba_dataset import CelebADataset


class ImageHelper(Helper):

    def poison(self):
        return

    
    def load_celeba_data(self):
        """Build and return a data loader."""
        self.name = self.params['name']
        
        crop_size = 178 #178 
        image_size = 128 #128 aka pixel
        
        flip = transforms.RandomHorizontalFlip()
        crop = transforms.CenterCrop(crop_size)
        resize = transforms.Resize(image_size)
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        
        #Cropping flipping and resizing image
        transform_train = transforms.Compose([flip, crop, resize, transforms.ToTensor(), normalize])
        #same as above but no flipping as it is the testing set
        # trace1
        transform_test = transforms.Compose([crop, resize, transforms.ToTensor(), normalize])

        #Accesed from celeba_dataset.py file trace1
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
        self.dataset_size = len(self.test_dataset)
        logger.info(f"Length of CelebA testing dataset: {self.dataset_size}")

        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.params['batch_size'],
                                                        shuffle=True, #Trace1
                                                        num_workers=4, # rule of thumb 4* GPU
                                                        pin_memory=True, #For Performance improvement
                                                        drop_last=True) #Drops last batch if not divisible by batch size

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=self.params['test_batch_size'],
                                                        shuffle=False, 
                                                        pin_memory=True, #For performance improvement
                                                        num_workers=4)
                
        self.labels = self.params['labels']
        return True