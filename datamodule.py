from    torch.autograd import Variable #, grad

class DataWrapper(object):
    def __init__(self, datapath, gpucount):
        super().__init__()
        self.datapath       = datapath
        self.gpucount       = gpucount
        self.epoch_iterator = None
        self.batch_size     = 0

    def next_batch(self):
        try:
            batch = next(self.epoch_iterator)
        except (OSError, StopIteration):
            # restart dataset if epoch ended
            self.epoch_iterator = self.reset_epoch(*self.epoch_args)
            batch = next(self.epoch_iterator)
        return [Variable(b).cuda(async=(self.gpucount > 1)) for b in batch]

    def init_epoch(self, *epoch_args):
        self.epoch_args     = epoch_args
        self.epoch_iterator = self.reset_epoch(*self.epoch_args)

    ##### Virtual functions #####
    def epoch_len(self):
        assert(0)

    def reset_epoch(self, *args):
        assert(0)

########## JUNK ##############

    # def get_loader(self, datapath, gpucount, *args, **kwargs):
    #     assert(0)
