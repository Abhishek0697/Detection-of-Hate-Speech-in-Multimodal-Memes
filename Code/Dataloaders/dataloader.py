import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils import data
from torchvision import transforms, datasets, models
from PIL import Image


'''
Load the BERT tokenizer.
'''
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)




'''
Dataloader for Training/Validation
Returns (Image, Caption, Input_id, Attention_mask, label)
'''
class mydataset():    

    def __init__(self, classification_list, name):

        super(mydataset).__init__()
        
        self.X = []
        self.Cap = []
        self.Y = []
        
        with open(classification_list, mode = 'r') as f:
            
            for line in f:
                path, caption, label = line[:-1].split('\t')

                self.X.append('/home/ironman/abhishek/GBM/FB/Code/data/'+path)
                self.Cap.append(caption)
                self.Y.append(label)
        
        '''
        Tokenize all of the captions and map the tokens to thier word IDs, and get respective attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.Cap)
        
        
        
        '''
        Image Transforms
        '''
        
        if name in ['valid','test']:
            self.transform = transforms.Compose([   transforms.Resize(384),
                                                 transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([ transforms.Resize(256),
                                                 transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                                                            ])
    
    
    def __getitem__(self,index):
        
        
        '''
        For Image and Label
        '''
        image = self.X[index]
                
        image = (Image.open(image))
               
        image = self.transform(image)
        
        label = float(self.Y[index])

        
        '''
        For Captions, Input ids and Attention mask
        '''
        caption = self.Cap[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        
        return image, caption, input_id, attention_masks, torch.as_tensor(label).long()
        
  
    def __len__(self):
        return len(self.X)
    
    
    
    
    
    
    
        
'''
tokenize all of the sentences and map the tokens to their word IDs.
'''

def tokenize(sequences):
    
    input_ids = []
    attention_masks = []

    # For every caption...
    for seq in sequences:
        '''
        `encode_plus` will:
          (1) Tokenize the caption.
          (2) Prepend the `[CLS]` token to the start.
          (3) Append the `[SEP]` token to the end.
          (4) Map tokens to their IDs.
          (5) Pad or truncate the sentence to `max_length`
          (6) Create attention masks for [PAD] tokens.
        '''
        encoded_dict = tokenizer.encode_plus(
                            seq,                       # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 48,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',      # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    
    return input_ids, attention_masks


'''
Toy example explaining the working of tokenize function and max_len=48

Original Caption: 
a phobia is an irrational fear a fear that muslims may be terrorists is not islamaophobia but a fear grounded in history, experience, and reality

Token IDs: tensor([  101,  1037,  6887, 16429,  2401,  2003,  2019, 23179,  3571,  1037,
         3571,  2008,  7486,  2089,  2022, 15554,  2003,  2025,  7025,  7113,
        24920,  2021,  1037,  3571, 16764,  1999,  2381,  1010,  3325,  1010,
         1998,  4507,   102,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0])
            
Attention masks: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

'''





'''
Dataloader for creating predictions.csv
Returns (Image, Captions, Input_id, Attention_mask and ImageName)
'''
class mytestdataset():    

    def __init__(self, classification_list, name):

        super(mytestdataset).__init__()
        
        self.X = []
        self.Cap = []
        self.Imagename = []
        
        with open(classification_list, mode = 'r') as f:
            
            for line in f:
                path, caption = line[:-1].split('\t')

                self.X.append('/home/ironman/abhishek/GBM/FB/Code/data/'+path)
                self.Cap.append(caption)
                self.Imagename.append(path.split('/')[1][:-4])
        
        
        '''
        Tokenize all of the captions and map the tokens to their word IDs, and get respective attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.Cap)
        
        
        
        '''
        Image Transforms
        '''
        self.transform = transforms.Compose([   transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                            ])
         
    
    def __getitem__(self,index):
        
        
        '''
        Image
        '''
        image = self.X[index]
                
        image = (Image.open(image))
               
        image = self.transform(image)
        
       
        '''
        For Captions, Input ids, Attention mask and Imagename
        '''
        caption = self.Cap[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        Imagename = self.Imagename[index]
        
        return image, caption, input_id, attention_masks, Imagename
        
  
    def __len__(self):
        return len(self.X)
    



    
    
    
    
    
    
    
    
    
       
'''
Dataloader for Training/Validation with support for Image Captioning model
Returns (Image, Caption, Input_id, Attention_mask, Input_id_Captioning_model, Attention_mask_Captioning_model, label)
'''
class mydataset_captioning():    

    def __init__(self, classification_list, name):

        super(mydataset_captioning).__init__()
        
        self.X = []
        self.true_Cap = []
        self.generated_Cap = []
        self.Y = []
        
        with open(classification_list, mode = 'r') as f:

            for line in f:
                
                path, caption, generated_caption, label = line[:-1].split('\t')

                self.X.append('/home/ironman/abhishek/GBM/FB/Code/data/'+path)
                self.true_Cap.append(caption)
                self.generated_Cap.append(generated_caption)
                self.Y.append(label)
        
        '''
        Tokenize all of the captions and map the tokens to thier word IDs, and get respective attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.true_Cap)
        
        self.input_ids_cap, self.attention_masks_cap = tokenize(self.generated_Cap)
        
        
        
        '''
        Image Transforms
        '''
        
        if name in ['valid','test']:
            self.transform = transforms.Compose([   transforms.Resize(384),
                                                 transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([ transforms.Resize(256),
                                                 transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                                                            ])
    
    
    def __getitem__(self,index):
        
        
        '''
        For Image and Label
        '''
        image = self.X[index]
                
        image = (Image.open(image))
               
        image = self.transform(image)
        
        label = float(self.Y[index])

        
        '''
        For Captions, Input ids and Attention mask
        '''
        caption = self.true_Cap[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
            
        input_id_cap = self.input_ids_cap[index]
        attention_masks_cap = self.attention_masks_cap[index]
    
            
            
        return image, caption, input_id, attention_masks, input_id_cap, attention_masks_cap, torch.as_tensor(label).long()
        
  
    def __len__(self):
        return len(self.X)
        
