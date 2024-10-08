import sys
import os 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.task_sampler import TaskSampler
from torch.utils.data import DataLoader
from datasets.miniimagenet import MiniImageNet
import torch
from datasets.task_sampler import TaskSampler
from torch.utils.data import DataLoader
from loops.training_loop import training_loop
from loops.testing_loop import testing_loop

GPU_INDEX = 1
device = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
print(device)

mini_imagenet_path = "../../../data/mini_imagenet/images"
train_set = MiniImageNet(root=mini_imagenet_path, split="train")
train_sampler = TaskSampler(train_set, n_way=5, n_shot=5, n_query= 15, n_tasks=100)
train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,

    )


validation_set = MiniImageNet(root = mini_imagenet_path, split = "val")
validation_sampler = TaskSampler(validation_set, n_way = 5, n_shot = 5, n_query = 15, n_tasks = 50)
validation_loader = DataLoader(validation_set, batch_sampler = validation_sampler, pin_memory = True, collate_fn = validation_sampler.episodic_collate_fn)
training_loop(train_loader= train_loader, 
              validation_loader = validation_loader,
               experiment_config = 'src/experiment_config.json',
               hyperparameter_m = 6, 
               hyperparameter_T = 1,
               hyperparameter_lambda = 0.1,
               encoder_weights = 'saved_encoders/encoder_epoch_146_val_acc_0.4419.pth',
               device = device)


test_set  = MiniImageNet(root = mini_imagenet_path , split = 'test')
test_sampler = TaskSampler(test_set, n_way = 5, n_shot = 5, n_query = 15, n_tasks = 200)
test_loader = DataLoader (test_set, batch_sampler = test_sampler , pin_memory = True , collate_fn = test_sampler.episodic_collate_fn)

testing_loop(test_loader = test_loader , experiment_config = 'src/experiment_config.json',device = 'cpu')
