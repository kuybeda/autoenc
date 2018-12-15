from    torch.autograd import Variable #, grad

class DataWrapper(object):
    def __init__(self, datapath):
        super().__init__()
        self.datapath       = datapath
        # self.gpucount       = gpucount
        self.epoch_iterator = None
        self.batch_size     = 0

    def next_batch(self, *args, **kwargs):
        try:
            batch = next(self.epoch_iterator)
        except (OSError, StopIteration):
            # restart dataset if epoch ended
            del self.epoch_iterator
            self.epoch_iterator = self.reset_epoch(*self.epoch_args)
            batch = next(self.epoch_iterator)

        batch = self.postproc_next_batch(batch, *args, **kwargs)
        return [b.cuda() for b in batch]

    def postproc_next_batch(self, batch, *args, **kwargs):
        return batch # default function does nothing

    def init_epoch(self, *epoch_args):
        self.epoch_args     = epoch_args
        if self.epoch_iterator is not None:
            del self.epoch_iterator
        self.epoch_iterator = self.reset_epoch(*self.epoch_args)

    ##### Virtual functions #####
    # def epoch_len(self):
    #     assert(0)

    def reset_epoch(self, *args):
        assert(0)

########## JUNK ##############

    # def stop_batches(self):
    #     del self.epoch_iterator

    # def get_loader(self, datapath, gpucount, *args, **kwargs):
    #     assert(0)
