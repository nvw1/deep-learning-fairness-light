# Based on https://github.com/FarrandTom/deep-learning-fairness
# from @article{farrand2020neither, title={Neither Private Nor Fair: Impact of Data Imbalance on Utility and Fairness in Differential Privacy}, author={Farrand, Tom and Mireshghallah, Fatemehsadat and Singh, Sahib and Trask, Andrew}, journal={arXiv preprint arXiv:2009.06389}, year={2020} }

import os
import random

from torch.utils.data import Dataset #This is important
from PIL import Image


class CelebADataset(Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attr, protected_attr, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attr = selected_attr
        self.protected_attr = protected_attr
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocessEqual()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file.
            This precossing version picks entries for training and testing set at random
        
        """

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split(',')
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        # Data pipeline trace1
        for i, line in enumerate(lines):
            split = line.split(',')
            filename = split[0]
            values = split[0:]
            idx = self.attr2idx[self.selected_attr]
            protected_idx = self.attr2idx[self.protected_attr]
            
            if values[idx] == '1':
                label = int(1)
            else:
                label = int(0)
               
            if values[protected_idx] == '1':
                protected_label = int(1)
            else:
                protected_label = int(0)

            # Use an 80/20 train/test split. With 30,000 in the dataset this is 6,000 in test. 
            if (i+1) < 6000:
                self.test_dataset.append([filename, protected_label, label])
            else:
                self.train_dataset.append([filename, protected_label, label])
            

        print('Finished preprocessing the CelebA dataset...')
    
    def preprocessEqual(self):
        """Preprocess the CelebA attribute file.
            This preprocessing method ensures the ratio of proteced and unprotected attributes in the test set reflect the ratios of the data set as a whole.
        """
        print('Entering equal preprocessing. Change setting in celeba_dataset.py')
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split(',')
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234) # Data pipline trace 1
        classOneMale = 0
        classTwoFemale = 0

        # Re counting the balance to allow for fair distribution
        for i, line in enumerate(lines):
            split = line.split(',')
            filename = split[0]
            values = split[0:]
            idx = self.attr2idx[self.selected_attr]
            protected_idx = self.attr2idx[self.protected_attr]
            if values[protected_idx] == '1':
                classOneMale += 1
            else:
                classTwoFemale += 1
        
        
        ratio = classOneMale/classTwoFemale
        print("The ratio male/female for this run  is:",ratio)

        #As the training set will be 20% of data we re do this here to ensure the right min amount later
        maleMin = classOneMale/5
        femaleMin = classTwoFemale/5
        maleCounter = 0
        femaleCounter = 0
        testSetCounter = 0 

        maleMinSmile = maleMin/2
        femaleMinSmile = femaleMin/2
        maleSmileCount = 0
        maleNoSmileCount = 0
        femaleSmileCount = 0
        femaleNoSmileCount = 0
        print("maleMin=",maleMin,"FemaleMin = ", femaleMin)
        

        for i, line in enumerate(lines):
            split = line.split(',')
            filename = split[0]
            values = split[0:]
            idx = self.attr2idx[self.selected_attr]
            protected_idx = self.attr2idx[self.protected_attr]
            
            if values[idx] == '1':
                label = int(1)
            else:
                label = int(0)
               
            if values[protected_idx] == '1':
                protected_label = int(1)
            else:
                protected_label = int(0)

            # Use an 80/20 train/test split. With 30,000 in the dataset this is 6,000 in test.
            if testSetCounter < 6000:
                if (protected_label == 1): # If male
                    if maleCounter < maleMin: # If we dont have enough males in training set
                        if label == 1: #If Smiling 
                            if maleSmileCount < maleMinSmile: # only get enough
                                self.test_dataset.append([filename, protected_label, label])
                                testSetCounter += 1
                                maleSmileCount += 1
                            else:
                                self.train_dataset.append([filename, protected_label, label])
                        else: # If male not smiling
                            if maleNoSmileCount < maleMinSmile: # only get enough
                                self.test_dataset.append([filename, protected_label, label])
                                testSetCounter += 1
                                maleNoSmileCount += 1
                            else:
                                self.train_dataset.append([filename, protected_label, label])
                    else: # Add male to training as there is enough in test data
                        self.train_dataset.append([filename, protected_label, label])

                else : # If female
                    if femaleCounter < femaleMin: # if there is not enought in tringinge
                        if label == 1: #If Smiling 
                            if femaleSmileCount < femaleMinSmile: # only get enough
                                self.test_dataset.append([filename, protected_label, label])
                                testSetCounter += 1
                                femaleSmileCount +=1
                            else:
                                self.train_dataset.append([filename, protected_label, label])
                        else: # If male not smiling
                            if femaleNoSmileCount < femaleMinSmile: # only get enough
                                self.test_dataset.append([filename, protected_label, label])
                                testSetCounter += 1
                                femaleNoSmileCount += 1
                            else:
                                self.train_dataset.append([filename, protected_label, label])
                    else: # Add male to training as there is enough in test data
                        self.train_dataset.append([filename, protected_label, label])
            else:
                self.train_dataset.append([filename, protected_label, label])

            
            
        print("maleSmileCount:",maleSmileCount,"maleNoSmileCount:",maleNoSmileCount,"femaleSmileCount:",femaleSmileCount,"femaleNoSmileCount:",femaleNoSmileCount)
        print('Finished preprocessing the CelebA dataset... the equal way')


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, protected_label, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), protected_label, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images