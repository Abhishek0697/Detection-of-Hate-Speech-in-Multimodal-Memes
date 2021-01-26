import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict


train_loss= []
v_loss = []
v_acc = []


def train(model, data_loader, test_loader, criterion, optimizer, lr_scheduler, modelpath, writer, device, epochs):
    
    model.train()

    for epoch in range(epochs):
        avg_loss = 0.0
                
        
        for batch_num, (feats, captions, input_id, attention_masks, target) in enumerate(data_loader):
            feats, target = feats.to(device), target.to(device)
            input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
                
            '''
            Compute output and loss from BERT
            '''
            loss, logits = model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=attention_masks, 
                             labels=target
                                )

                   
            '''
            Take Step
            '''                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            '''
            Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            
            avg_loss += loss.item()


            '''
            linear_schedule_with_warmup take step after each batch
            '''
            lr_scheduler.step()
            
                        
            if batch_num % 100 == 99:
                print('loss', avg_loss/100)
                
            del feats
            del captions
            del input_ids
            del attention_masks
            del target
            del loss
            
            
        training_loss = avg_loss/len(data_loader)
       
        print('Epoch: ', epoch+1)            
        print('training loss = ', training_loss)
        train_loss.append(training_loss)


        '''
        Check performance on validation set after an Epoch
        '''

        valid_loss, top1_acc= test_classify(model, test_loader, criterion, device)
        print('Validation Loss: {:.4f}\tTop 1 Validation Accuracy: {:.4f}'.format(valid_loss, top1_acc))
        v_loss.append(valid_loss)
        v_acc.append(top1_acc)

    
        writer.add_scalar("Loss/train", training_loss, epoch)            
        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        writer.add_scalar('Accuracy/Validation', top1_acc, epoch)
       
        
        '''
        save model checkpoint after every epoch
        '''

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training_Loss_List':train_loss,
            'Validation_Loss_List':v_loss,
            'Validation_Accuracy_List': v_acc,
            'Epoch':epoch,
            'lr_scheduler': lr_scheduler.state_dict() 

            }, modelpath)


        
'''
Returns Loss, top1 and top5 accuracies on test/validation set
'''
def test_classify(model, test_loader, criterion, device):
    model.eval()
    test_loss = []
    top1_accuracy = 0
    total = 0

    for batch_num, (feats, captions, input_id, attention_masks, target) in enumerate(test_loader):
        
        feats, target = feats.to(device), target.to(device)
        input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
            
        '''
        Compute output and loss from BERT
        '''
        loss, logits = model(input_ids, 
                         token_type_ids=None, 
                         attention_mask=attention_masks, 
                         labels=target)

        test_loss.extend([loss.item()]*feats.size()[0])
        
        predictions = F.softmax(logits, dim=1)
        
        _, top1_pred_labels = torch.max(predictions,1)
        top1_pred_labels = top1_pred_labels.view(-1)
        
        top1_accuracy += torch.sum(torch.eq(top1_pred_labels, target)).item()
        
        
        total += len(target)
        
        del feats
        del captions
        del input_ids
        del attention_masks
        del target
        del loss
            
    model.train()
    return np.mean(test_loss), top1_accuracy/total
