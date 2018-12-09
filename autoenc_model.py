import  torch
from    torch import nn,optim
# from    torch.nn import functional as F
import  utils
import  config
from    model_utils import Generator, Encoder, Critic
from    torch.autograd import Variable, grad
# from    cnnutils import EqualConv2d, PixelNorm, DenseBlock, SpectralNormConv2d

args   = config.get_config()

def critic_grad_penalty(critic, x, fake_x, batch, phase, alpha, grad_norm):
    eps         = torch.rand(batch, 1, 1, 1).cuda()
    x_hat       = eps * x.data + (1 - eps) * fake_x.data
    x_hat       = Variable(x_hat, requires_grad=True)
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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = nn.DataParallel(Encoder(args.nz).cuda())
        self.generator  = nn.DataParallel(Generator(args.nz).cuda())
        self.critic     = nn.DataParallel(Critic(args.nz).cuda())
        self.reset_optimization()

    def reset_optimization(self):
        self.optimizerG = optim.Adam(self.generator.parameters(), args.EGlr, betas=(0.0, 0.99))
        self.optimizerE = optim.Adam(self.encoder.parameters(), args.EGlr, betas=(0.0, 0.99))
        self.optimizerC = optim.Adam(self.critic.parameters(), args.Clr, betas=(0.0, 0.99))

    def update(self, batch, phase, alpha):
        encoder, generator, critic = self.encoder, self.generator, self.critic
        stats, losses = {}, []
        utils.requires_grad(encoder, True)
        utils.requires_grad(generator, True)
        utils.requires_grad(critic, False)
        encoder.zero_grad()
        generator.zero_grad()

        x = batch[0]
        batch_size = x.shape[0]

        real_z = encoder(x, phase, alpha)
        fake_x = generator(real_z, phase, alpha)

        # use no gradient propagation if no x metric is required
        if args.use_x_metric:
            # match x: E_x||g(e(x)) - x|| -> min_e
            err_x = utils.mismatch(fake_x, x, args.match_x_metric)
            losses.append(err_x)
        else:
            with torch.no_grad():
                err_x = utils.mismatch(fake_x, x, args.match_x_metric)
        stats['x_err'] = err_x

        if args.use_z_metric:
            # cyclic match z E_x||e(g(e(x))) - e(x)||^2
            fake_z = encoder(fake_x, phase, alpha)
            err_z = utils.mismatch(real_z, fake_z, args.match_z_metric)
            losses.append(err_z)
        else:
            with torch.no_grad():
                fake_z = encoder(fake_x, phase, alpha)
                err_z = utils.mismatch(real_z, fake_z, args.match_z_metric)
        stats['z_err'] = err_z

        cls_fake = critic(fake_x, x, phase, alpha)

        cls_real = critic(x, x, phase, alpha)

        # measure loss only where real score is highre than fake score
        G_loss = -(cls_fake * (cls_real.detach() > cls_fake.detach()).float()).mean()

        # Gloss      = -torch.log(cls_fake).mean()
        stats['G_loss'] = G_loss
        # warm up critic loss to kick in with alpha
        losses.append(alpha * G_loss)

        # Propagate gradients for encoder and decoder
        loss = sum(losses)
        loss.backward()

        # Apply encoder and decoder gradients
        self.optimizerE.step()
        self.optimizerG.step()

        ###### Critic ########
        losses = []
        utils.requires_grad(critic, True)
        utils.requires_grad(encoder, False)
        utils.requires_grad(generator, False)
        critic.zero_grad()
        # Use fake_x, as fixed data here
        fake_x = fake_x.detach()

        cls_fake = critic(fake_x, x, phase, alpha)
        cls_real = critic(x, x, phase, alpha)

        cf, cr = cls_fake.mean(), cls_real.mean()
        C_loss = cf - cr + torch.abs(cf + cr)

        grad_norm = autoenc_grad_norm(encoder, generator, x, phase, alpha).mean()
        grad_loss = critic_grad_penalty(critic, x, fake_x, batch_size, phase, alpha, grad_norm)
        stats['grad_loss'] = grad_loss
        losses.append(grad_loss)

        # C_loss      = -torch.log(1.0 - cls_fake).mean() - torch.log(cls_real).mean()

        stats['cls_fake'] = cls_fake.mean()
        stats['cls_real'] = cls_real.mean()
        stats['C_loss'] = C_loss.data

        # Propagate critic losses
        losses.append(C_loss)
        loss = sum(losses)
        loss.backward()

        # Apply critic gradient
        self.optimizerC.step()
        return stats

    def pbar_description(self,stats,batch,batch_count,sample_i,phase,alpha,res,epoch):
        xr = stats['x_err']
        return ('{0}; it: {1}; phase: {2}; batch: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; x-err {7:.4f};').format(\
                batch_count+1, sample_i+1, phase, batch, alpha, res, epoch, xr)

    def dry_run(self, batch, phase, alpha):
        ''' dry run model on the batch '''
        encoder,generator = self.encoder,self.generator
        generator.eval()
        encoder.eval()
        x = batch[0]
        batch_size = x.shape[0]

        utils.requires_grad(generator, False)
        utils.requires_grad(encoder, False)

        real_z      = encoder(x, phase, alpha)
        reco_ims    = generator(real_z, phase, alpha).data

        # join source and reconstructed images side by side
        out_ims     = torch.cat((x,reco_ims), 1).view(2*batch_size,1,reco_ims.shape[-2],reco_ims.shape[-1])

        # utils.requires_grad(generator, True)
        # utils.requires_grad(encoder, True)
        encoder.train()
        generator.train()
        return out_ims


# ############# JUNK #########################

    # def save(self,path):
    #     torch.save(self.state_dict(),path)
        # torch.save({'G_state_dict': self.generator.state_dict(),
        #             'E_state_dict': self.encoder.state_dict(),
        #             'C_state_dict': self.critic.state_dict(),
        #             'optimizerE': self.optimizerE.state_dict(),
        #             'optimizerG': self.optimizerG.state_dict(),
        #             'optimizerC': self.optimizerC.state_dict()}, path)

    # def load(self, path):
    #     self.load_state_dict()


        # self.to_gray = nn.ModuleList([nn.Conv2d(nz, 1, 1), #Each has 1 channel and kernel size 1x1!
        #                              nn.Conv2d(nz, 1, 1),
        #                              nn.Conv2d(nz, 1, 1),
        #                              nn.Conv2d(nz, 1, 1),
        #                              nn.Conv2d(int(nz/2), 1, 1),
        #                              nn.Conv2d(int(nz/4), 1, 1),
        #                              nn.Conv2d(int(nz/8), 1, 1),
        #                              nn.Conv2d(int(nz/16), 1, 1),
        #                              nn.Conv2d(int(nz/32), 1, 1)])


        # self.from_gray = nn.ModuleList([nn.Conv2d(in_channels, int(nz/32), 1),
        #                                nn.Conv2d(in_channels, int(nz/16), 1),
        #                                nn.Conv2d(in_channels, int(nz/8), 1),
        #                                nn.Conv2d(in_channels, int(nz/4), 1),
        #                                nn.Conv2d(in_channels, int(nz/2), 1),
        #                                nn.Conv2d(in_channels, nz, 1),
        #                                nn.Conv2d(in_channels, nz, 1),
        #                                nn.Conv2d(in_channels, nz, 1),
        #                                nn.Conv2d(in_channels, nz, 1)])


        # # same as encoder with detection layer
        # self.classifier  = nn.Sequential(nn.LeakyReLU(0.2),
        #                                  EqualConv2d(nz, nz, 3),
        #                                  PixelNorm(),
        #                                  nn.LeakyReLU(0.2),
        #                                  nn.AdaptiveAvgPool2d(1),
        #                                  EqualConv2d(nz, 1, 1))


# class Attention(Generator):
#     ''' Mask generator that focuses on best discrepancy region '''
#     def __init__(self, nz, pixel_norm=False, spectral_norm=True):
#         super().__init__(nz, pixel_norm, spectral_norm)
#         self.to_nz = nn.Sequential(EqualConv2d(2*nz, nz, 1),nn.LeakyReLU(0.2))
#
#     def forward(self, input1, input2, step=0, alpha=-1):
#         input = self.to_nz(torch.cat((input1,input2),dim=1))
#         out   = super().forward(input, step, alpha)
#
#         alpha = 0.1
#         # Avoid zeros in mask
#         mask  = (torch.sigmoid(out) + alpha)/(1.0+alpha)
#         return mask


        # normalize max value to 1
        # mx,_ = mask.max(dim=-2,keepdim=True)
        # mx,_ = mx.max(dim=-1,keepdim=True)

        # nn.Sequential(
        # Conv2dNorm('conv1', nz, nz, 4, 3,
        #            pixel_norm=pixel_norm, spectral_norm=spectral_norm),

        # return self.classifier(out).squeeze(2).squeeze(2)

        # self.classifier  = nn.Sequential(EqualConv2d(nz, nz, 1, padding=0), PixelNorm(), nn.LeakyReLU(0.2),
        #                                  nn.Conv2d(nz, 1, 1))

                                          # nn.Sequential(
                                          # Conv2dNorm('conv1', nz, nz, 4, 0,
                                          #            pixel_norm=pixel_norm, spectral_norm=spectral_norm))])
        #
        # self.progression = nn.ModuleList([  ConvBlock(int(nz / 32), int(nz / 16), 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(int(nz / 16), int(nz / 8), 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(int(nz / 8), int(nz / 4), 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(int(nz / 4), int(nz / 2), 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(int(nz / 2), nz, 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(nz, nz, 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(nz, nz, 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(nz, nz, 3, 1,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                     ConvBlock(nz, nz, 3, 1, 4, 0,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm)])


        # self.progression = nn.ModuleList([ConvBlock(nz, nz, 4, 3, 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(nz, nz, 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(nz, nz, 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(nz, nz, 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(nz, int(nz/2), 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(int(nz/2), int(nz/4), 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(int(nz/4), int(nz/8), 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(int(nz/8), int(nz/16), 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm),
        #                                   ConvBlock(int(nz/16), int(nz/32), 3, 1,
        #                                             pixel_norm=pixel_norm, spectral_norm=spectral_norm)])


# class Critic(nn.Module):
#     def __init__(self, nz):
#         super().__init__()
#
#         self.classifier = nn.Sequential(EqualConv2d(nz, nz, 3, padding=0), PixelNorm(), nn.LeakyReLU(0.2),
#                                         EqualConv2d(nz, nz, 3, padding=0), PixelNorm(), nn.LeakyReLU(0.2),
#                                         nn.AdaptiveAvgPool2d(1),
#                                         nn.Conv2d(nz, 1, 1))
#
#     def forward(self, input):
#         return torch.sigmoid(self.classifier(input).squeeze(2)).squeeze(2)

        # out = utils.normalize(out)


        # self.classifier  = nn.Sequential(SpectralNormConv2d(nz,nz,3,padding=0), nn.LeakyReLU(0.2),
        #                                 SpectralNormConv2d(nz, nz, 3, padding=0),nn.LeakyReLU(0.2),
        #                                 nn.AdaptiveAvgPool2d(1),
        #                                 nn.Conv2d(nz, 1, 1))



        # if self.binary_predictor:
        #     self.linear = nn.Linear(nz, 1 + n_label)


        # if self.binary_predictor:
        #     out = self.linear(z_out)
        #     return out[:, 0], out[:, 1:]
        # else:

        # print(input.size(), out.size(), step)

            # if i == 0 and use_mean_std_layer:
            #     mean_std = input.std(0).mean()
            #     mean_std = mean_std.expand(input.size(0), 1, 4, 4)
            #     out = torch.cat([out, mean_std], 1)

# def init_linear(linear):
#     init.xavier_normal(linear.weight)
#     linear.bias.data.zero_()
#
#
# def init_conv(conv, glu=True):
#     init.kaiming_normal(conv.weight)
#     if conv.bias is not None:
#         conv.bias.data.zero_()

            # if use_ALQ == 1:
            #     print('Reserved for future use')

        # self.label_embed = nn.Embedding(n_label, n_label)
        # self.label_embed.weight.data.normal_()


#        import ipdb
#        ipdb.set_trace()

# input = self.code_norm(input)
# ARI: THis causes internal assertion failure. Maybe we don't need the embedding?
# label = self.label_embed(label)


# self.to_rgb = nn.ModuleList([nn.Conv2d(nz, 3, 1), #Each has 3 out channels and kernel size 1x1!
        #                              nn.Conv2d(nz, 3, 1),
        #                              nn.Conv2d(nz, 3, 1),
        #                              nn.Conv2d(nz, 3, 1),
        #                              nn.Conv2d(int(nz/2), 3, 1),
        #                              nn.Conv2d(int(nz/4), 3, 1),
        #                              nn.Conv2d(int(nz/8), 3, 1),
        #                              nn.Conv2d(int(nz/16), 3, 1),
        #                              nn.Conv2d(int(nz/32), 3, 1)])

        # self.from_rgb = nn.ModuleList([nn.Conv2d(3, int(nz/32), 1),
        #                                nn.Conv2d(3, int(nz/16), 1),
        #                                nn.Conv2d(3, int(nz/8), 1),
        #                                nn.Conv2d(3, int(nz/4), 1),
        #                                nn.Conv2d(3, int(nz/2), 1),
        #                                nn.Conv2d(3, nz, 1),
        #                                nn.Conv2d(3, nz, 1),
        #                                nn.Conv2d(3, nz, 1),
        #                                nn.Conv2d(3, nz, 1)])
