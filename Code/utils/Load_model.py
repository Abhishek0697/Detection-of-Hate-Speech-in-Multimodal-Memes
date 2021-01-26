import torch

######### Load saved model from checkpoint  #########

def load(modelpath, model, optimizer, lr_scheduler):
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss = checkpoint['Training_Loss_List'] 
    v_loss = checkpoint['Validation_Loss_List']
    v_acc = checkpoint['Validation_Accuracy_List']

    epoch = checkpoint['Epoch']
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    return model, optimizer, lr_scheduler, train_loss, v_loss, v_acc, epoch
