from    torch.autograd import Variable #, grad

class DataWrapper(object):
    def __init__(self, datapath, gpucount):
        super().__init__()
        self.datapath       = datapath
        self.gpucount       = gpucount
        self.loader         = self.get_loader(datapath, gpucount)
        self.epoch_iterator = None
        self.batch_size     = 0

    def next_batch(self):
        try:
            batch, _ = next(self.epoch_iterator)
        except (OSError, StopIteration):
            # restart dataset if epoch ended
            self.epoch_iterator = self.reset_epoch(*self.epoch_args)
            batch,_ = next(self.epoch_iterator)
        return Variable(batch).cuda(async=(self.gpucount > 1))

    def init_epoch(self, *epoch_args):
        self.epoch_args     = epoch_args
        self.epoch_iterator = self.reset_epoch(*self.epoch_args)

    def epoch_len(self):
        return len(self.loader(1,4).dataset)

    ##### Virtual functions #####
    def get_loader(self, datapath, gpucount, *args, **kwargs):
        assert(0)

    def reset_epoch(self, *args):
        assert(0)


