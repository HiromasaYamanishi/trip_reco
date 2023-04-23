import matplotlib.pyplot as plt
import numpy as np
import torch

def save_plot(epoch_num, name, **kwargs):
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots()
    epoch = [i+1 for i in range(epoch_num)]
    c = ['blue','green','orange','cyan','red','yellow']
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel(name, fontsize=18)
    ax.set_title(name, fontsize=18)
    for i,(k,v) in enumerate(kwargs.items()):
        ax.plot(epoch, v, color=c[i], label=k)
    ax.legend(loc=0, fontsize=18)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    plt.savefig(f'{name}.png') # 画像の保存


def save_cor(x, y, x_name, y_name,save_name,*args):
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    cor=np.corrcoef(x, y)[0][1]
    ax.set_title(f'cor: {round(cor,5)}', fontsize=18)
    ax.scatter(x, y)
    fig.subplots_adjust(bottom = 0.15)
    plt.savefig(f'{save_name}.png')
    if len(args)>0:
        f = open('out_spot.txt', 'w')
        for i, arg in enumerate(args[0]):
            if abs(x[i]-y[i])>1.5:
                ax.annotate(arg, (x[i],y[i]), fontname='Noto Serif CJK JP')
                f.write(arg)
                f.write(f' gt:{x[i]}')
                f.write(f' pred:{y[i]}')
                f.write('\n')
        f.close()
        plt.savefig('cor_with_name.png')
    #plt.savefig(f'{save_name}.png')

def write_cor(cor):
    with open('result.txt','a') as f:
        f.write(str(cor))
        f.write('\n')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

