################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name   = 'sgd_pre_da' # name of experiment

# Model Options
model_type = 'mynet' # 'mynet' or 'resnet18'

# Learning Options
epochs     = 100           # train how many epochs
batch_size = 128           # batch size for dataloader 
use_adam   = False        # Adam or SGD optimizer
lr         = 1e-2         # learning rate
milestones = [50, 80] # reduce learning rate at 'milestones' epochs
