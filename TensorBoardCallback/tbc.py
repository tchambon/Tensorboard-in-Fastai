from torch.utils.tensorboard import SummaryWriter
from fastai.vision import *

class TensorBoardFastAI(LearnerCallback):
    def __init__(self, writer, learn, track_weight=False, track_grad=False, metric_names=['Validation loss', 'Accuracy']):
        """
        Create a callback compatible with fastai

        track_weight is used to decide if weights will be logged and displayed as histograms in Tensorboard
        track_grad is used to decide if gradients will be logged and displayed as histograms in Tensorboard
        metric_names are the names to be displayed in Tensorboard. The first one is always validation loss
            The order has to be the same than in learn.metrics
        """

        self.writer=writer
        self.learn = learn
        self.last_epoch_backward = -1
        self.track_weight = track_weight
        self.track_grad = track_grad
        self.metric_names = metric_names
        
    def on_epoch_end(self, **kwargs:Any):
        # Do not update stats if we are in the lr_finder case
        if kwargs['last_metrics'][0] != None:
            
            self.writer.add_scalar('Loss', kwargs['last_loss'], kwargs['epoch'])
            for i, met in enumerate(kwargs['last_metrics']):
                self.writer.add_scalar(self.metric_names[i], met, kwargs['epoch'])
               
            
            
            if self.track_weight:
                for k, v in self.learn.model.state_dict().items():
                    self.writer.add_histogram(k, v, kwargs['epoch'])
            
                  
    def on_backward_end(self, **kwargs:Any):
        if self.track_grad and 'last_metrics' in kwargs and kwargs['last_metrics'][0] != None and kwargs['epoch'] > self.last_epoch_backward:
            self.last_epoch_backward = kwargs['epoch']
            keys = list(self.learn.model.state_dict().keys())
            indice = 0
            for i, param in enumerate(self.learn.model.parameters()):
                grad = param.grad
                if grad is not None:
                    self.writer.add_histogram('back'+'.'+keys[i], grad, kwargs['epoch'])
                    i += 1
