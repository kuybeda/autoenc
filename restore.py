import  torch
import  config
import  utils
import  evaluate
from    session import Session
from    torch.autograd import Variable, grad
from    torch import nn, optim
from    myplotlib import show_planes,imshow,clf
import  numbers

import  torch.backends.cudnn as cudnn
cudnn.benchmark = True
args     = config.get_config()

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.weight.shape[-1]//2)

def critic_grad_penalty(critic, x, fake_x, batch, phase, alpha, grad_norm):
    eps     = torch.rand(batch, 1, 1, 1).cuda()
    x_hat   = eps * x.data + (1 - eps) * fake_x.data
    x_hat   = Variable(x_hat, requires_grad=True)
    hat_predict = critic(x_hat, x, phase, alpha)
    grad_x_hat  = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
    # Push the gradients of the interpolated samples towards 1
    grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - grad_norm)**2).mean()
    return grad_penalty

def autoenc_grad_norm(encoder, generator, x, phase, alpha):
    x_hat        = Variable(x, requires_grad=True)
    z_hat        = encoder(x_hat, phase, alpha)
    fake_x_hat   = generator(z_hat, phase, alpha)
    err_x_hat    = utils.mismatch(fake_x_hat, x_hat, args.match_x_metric)
    grad_x_hat   = grad(outputs=err_x_hat.sum(), inputs=x_hat, create_graph=True)[0]
    grad_norm    = grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1)
    return grad_norm

def updateImages(input, session):
    encoder, generator, critic = session.encoder, session.generator, session.critic
    # batch,alpha,phase = session.cur_batch(), session.alpha, session.phase
    phase = 5
    alpha = 1.0

    stats  = {}
    utils.requires_grad(encoder, False)
    utils.requires_grad(generator, False)
    utils.requires_grad(critic, False)

    smoothing   = GaussianSmoothing(1, 11.0, 10.0).cuda()
    x           = smoothing(input)
    x           = torch.abs(x - x[[1]])#+ 0.5*torch.rand_like(input)
    x           = Variable(x, requires_grad=True)

    optimizer   = optim.Adam([x], 0.001, betas=(0.0, 0.99))

    while True:
        for i in range(1):
            losses = []
            optimizer.zero_grad()

            real_z      = encoder(x, phase, alpha)
            fake_x      = generator(real_z, phase, alpha)

            err_x       = utils.mismatch(fake_x, x, args.match_x_metric)
            losses.append(err_x)
            stats['x_err'] = err_x.data

            cls_fake    = critic(fake_x, x, session.phase, session.alpha)
            # measure loss only where real score is highre than fake score
            cls_loss    = -cls_fake.mean()
            stats['cls_loss']  = cls_loss.data
            # warm up critic loss to kick in with alpha
            losses.append(cls_loss)

            # Propagate gradients for encoder and decoder
            loss = sum(losses)
            loss.backward()

            g = x.grad.cpu().data

            # Apply encoder and decoder gradients
            optimizer.step()


        idx = 0
        imshow(x[idx,0].cpu().data)
        imshow(fake_x[idx,0].cpu().data)
        # imshow(input[idx,0].cpu().data)
        # imshow(g[0,0].cpu().data)

        clf()

    return stats

def train(session):
    # init progress bar and tensorboard statistics
    session.init_stats()
    # decide initial phase, alpha, etc
    session.init_phase()

    while session.sample_i < session.total_steps:
        # get data
        x     = session.get_next_batch()

        # update networks
        stats = updateImages(x, session)

        # show and save statistics to tensorboard
        session.handle_stats(stats)
        # decide phase, alpha, etc
        session.maintain_phase()
        # save checkpoint if time is right
        session.save_checkpoint()

        # write test images
        evaluate.tests_run(session, session.train_data_loader, reconstruction = (session.batch_count % 100 == 0))

    session.finish()

def main():
    session = Session()
    session.load_checkpoint()

    print('PyTorch {}'.format(torch.__version__))
    train(session)

if __name__ == '__main__':
    main()

