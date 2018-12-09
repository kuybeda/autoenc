import  torch
from    torch.utils.data import DataLoader
from    datamodule import DataWrapper
from    torchvision import datasets, transforms, utils
import  os
from    PIL import Image
from    custom_transform import NRandomCrop
from    torchvision.transforms import functional as F
import  multiprocessing
import  imageio
import  numpy as np
from    myplotlib import show_planes,imshow,clf

def mix_batch(batch,transform,alpha):
    batch_prev = torch.stack([transform(im[0][..., None].numpy()) for im in batch])
    return batch * alpha + (1.0 - alpha) * batch_prev

def collate_fn(batch):
    tlist = []
    for b in batch:
        # take only images and ommit lables
        for im in b[0]:
            tlist.append(F.to_tensor(im))
    return torch.cat(tlist,dim=0)[:,None]

class PairDataWrapper(DataWrapper):
    ''' Data wrapper for RGB and FIR image pairs '''
    def __init__(self, datapath, gpucount):
        super().__init__(datapath, gpucount)
        ncores = multiprocessing.cpu_count()
        self.loaders = PairDataWrapper.get_loaders(datapath, gpucount, num_workers=ncores)

    ## implementation of abstract functions
    @staticmethod
    def get_loaders(path, gpucount, num_workers, *args, **kwargs):
        def loaders(transform, batch_size):

            def png_reader(fname):
                im  = np.float32(imageio.imread(fname)) # 640x480
                im  = im[:400] # 640 x 400
                im -= im.mean()
                impl = Image.fromarray(im/8192.0) # convert to PIL with range roughy [-1,1]
                return impl.resize((320,200),Image.BILINEAR)

            def rgb_reader(fname):
                im   = np.float32(imageio.imread(fname)) # 1280 x 800
                im   = np.dot(im[...,:3], [0.299, 0.587, 0.114]) # to grayscale
                im  -= im.mean()
                impl = Image.fromarray(im/128.0)
                return impl.resize((320,200),Image.BILINEAR) # roughly to [-1,1]

            rgb_set     = datasets.DatasetFolder(os.path.join(path, 'RGB'), loader=rgb_reader,
                                                 extensions=['.jpg'], transform=transform)

            fir_set     = datasets.DatasetFolder(os.path.join(path, 'FIR'), loader=png_reader,
                                                 extensions = ['.png'], transform=transform)

            rgb_loader  = DataLoader(rgb_set, shuffle=True, batch_size=batch_size,
                                     num_workers=num_workers, drop_last=True,
                                     pin_memory=(gpucount>1),collate_fn=collate_fn)

            fir_loader  = DataLoader(fir_set, shuffle=True, batch_size=batch_size,
                                     num_workers=num_workers, drop_last=True,
                                     pin_memory=(gpucount>1),collate_fn=collate_fn)

            return {'RGB':rgb_loader,'FIR':fir_loader}
        return loaders

    def reset_epoch(self, batch_size, alpha, res, phase):
        N_RANDOM_CROPS = 8
        transform = transforms.Compose([
            NRandomCrop(size=128,n=N_RANDOM_CROPS),
            transforms.Lambda(lambda crops: [F.resize(crop,res,Image.BILINEAR) for crop in crops] )])

        loaders = self.loaders(transform, batch_size=batch_size//N_RANDOM_CROPS)

        for batch in zip(loaders['RGB'],loaders['FIR']):
            if alpha == 1.0 or phase == 0:
                yield batch
            else:
                tprev = transforms.Compose([
                    transforms.ToPILImage(mode='F'),
                    # Downsample
                    transforms.Resize(res // 2, Image.BILINEAR),
                    # Upsample with linear interpolation
                    transforms.Resize(res, interpolation=Image.BILINEAR), #Image.NEAREST
                    transforms.ToTensor(),
                ])
                yield [mix_batch(b,tprev,alpha) for b in batch]
        print("\nFinished epoch !! ")

    #### implementation of abstract functions
    def epoch_len(self):
        loaders = self.loaders(1,4)
        return min(len(loaders['RGB'].dataset),len(loaders['FIR'].dataset))

############ JUNK ################################
# transforms.functional.to_grayscale,
# tresize = transforms.Compose([transforms.Resize(res),transforms.ToTensor()])

# if __name__ == '__main__':
#     # flatten dataset subdirectory structure
#     from filetools import  link_dirtree, mkdir_assure
#
#     srclist = ['/data/domain_translation/RGB',
#                '/data/domain_translation/FIR']
#
#     extlist = ['.jpg','.png']
#
#     dstlist = ['/data/domain_translation/train/RGB',
#                '/data/domain_translation/train/FIR']
#
#     for s,d,e in zip(srclist,dstlist,extlist):
#         mkdir_assure(d)
#         link_dirtree(s,d,e,e)


# img_prev  = torch.stack([transform_prev(im[0][...,None].numpy()) for im in img])
# mixed_img = img * alpha + (1.0-alpha)*img_prev
# mix both RGB and FIR

# transform = transforms.Compose([
        #     # transforms.ToTensor(),
        #     # transforms.Lambda(lambda x:transforms.functional.to_grayscale(x)),
        #     # transforms.ToPILImage(),
        #     transforms.functional.to_grayscale,
        #     NRandomCrop(size=128,n=N_RANDOM_CROPS),
        #     transforms.Lambda(lambda crops: [F.resize(crop,res) for crop in crops] )
        #     # transforms.RandomCrop(128),
        #     # transforms.Resize(res),
        #     # transforms.ToTensor(),
        # ])




