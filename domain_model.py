from    model_utils import Generator, Encoder, Critic, OptimModule
import  config
import  torch
from    torch import nn,optim
import  utils
from    torch.autograd import Variable, grad
# import  numpy as np

args   = config.get_config()

class DualEncoder(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.rgb    = nn.DataParallel(Encoder(nz,pixel_norm=False, spectral_norm=True, make_shortcuts=True).cuda())
        self.fir    = nn.DataParallel(Encoder(nz,pixel_norm=False, spectral_norm=True, make_shortcuts=True).cuda())

class DualGenerator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.rgb    = nn.DataParallel(Generator(nz, pixel_norm=False, spectral_norm=True, use_shortcuts = True).cuda())
        self.fir    = nn.DataParallel(Generator(nz, pixel_norm=False, spectral_norm=True, use_shortcuts = True).cuda())

class DualCritic(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.rgb    = nn.DataParallel(Critic(nz, pixel_norm=False, spectral_norm=True).cuda())
        self.fir    = nn.DataParallel(Critic(nz, pixel_norm=False, spectral_norm=True).cuda())

#  don't inherit from nn.Module to avoid params double reference
class Autoencoder(object):
    def __init__(self, enc, gen):
        super().__init__()
        self.enc = enc
        self.gen = gen
    def __call__(self,x,phase,alpha):
        z,shorts = self.enc(x, phase, alpha)
        return self.gen(z, phase, alpha, shorts)

class DualAutoencoder(object):
    def __init__(self, dualenc, dualgen):
        super().__init__()
        self.rgb = Autoencoder(dualenc.rgb,dualgen.rgb)
        self.fir = Autoencoder(dualenc.fir,dualgen.fir)

class MixCoder(object):
    def __init__(self, enc1, enc2, gen1, gen2, style_channels):
        super().__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.gen1 = gen1
        self.gen2 = gen2
        # proportion of channels for style encoding
        self.style_channels = style_channels

    def split_channels(self,z):
        rchannels = self.style_channels
        return z[:,:rchannels], z[:,rchannels:]

    def rand_combine(self,z):
        z_style, z_struct = self.split_channels(z)
        # generate random style
        r_style = Variable(torch.randn_like(z_style), requires_grad=False)
        # normalize random component
        zsnorm  = z_style.norm(2, keepdim=True, dim=1)
        r_style = zsnorm*r_style/r_style.norm(2, keepdim=True, dim=1)
        return torch.cat((r_style,z_struct),dim=1)

    def cross_combine(self,z1,z2):
        z1_style, z1_struct  = self.split_channels(z1)
        _, z2_struct = self.split_channels(z2)
        # calculate difference in structure features
        # zerr         = z1_struct - z2_struct
        return torch.cat((z1_style,z2_struct),dim=1), z1_struct, z2_struct

    def __call__(self, x1, x2, phase, alpha):
        z1,short1 = self.enc1(x1, phase, alpha)
        # combine with random style input
        # z2      = self.rand_combine(z1)
        z2,_      = self.enc2(x2, phase, alpha)
        z2,_,_    = self.cross_combine(z2,z1)

        x2_hat  = self.gen2(z2, phase, alpha, short1)
        z2_hat,short2_hat = self.enc2(x2_hat, phase, alpha)
        # make a style shortcut from z1 to z1_hat
        z1_hat, z1_struct, z1_hat_struct = self.cross_combine(z1,z2_hat)
        x1_hat  = self.gen1(z1_hat, phase, alpha, short2_hat)
        return x2_hat, x1_hat, z1_struct, z1_hat_struct

class DualMixCoder(object):
    def __init__(self, dualenc, dualgen, style_channels):
        super().__init__()
        self.rgb = MixCoder(dualenc.rgb, dualenc.fir, dualgen.rgb, dualgen.fir, style_channels)
        self.fir = MixCoder(dualenc.fir, dualenc.rgb, dualgen.fir, dualgen.rgb, style_channels)

class Model(OptimModule):
    def __init__(self):
        super().__init__(args.load_optimizers)
        nz           = args.nz
        # style_prop   = 0.0625 # proportion of channels for style

        self.enc     = DualEncoder(nz)
        self.gen     = DualGenerator(nz)
        self.crt     = DualCritic(nz)
        # param manipulators, don't hold own params
        self.autoenc = DualAutoencoder(self.enc, self.gen)
        self.mixcod  = DualMixCoder(self.enc, self.gen, args.style_channels)
        self.reset_optimizers()

    def reset_optimizers(self):
        # self.optimE = optim.Adam(self.enc.parameters(), args.EGlr, betas=(0, 0.99))
        # self.optimG = optim.Adam(self.gen.parameters(), args.EGlr, betas=(0, 0.99))
        # self.optimC = optim.Adam(self.crt.parameters(), args.Clr,  betas=(0, 0.99))
        self.optimE = optim.Adamax(self.enc.parameters(), args.EGlr, betas=(0, 0.99), weight_decay=args.weight_decay)
        self.optimG = optim.Adamax(self.gen.parameters(), args.EGlr, betas=(0, 0.99), weight_decay=args.weight_decay)
        self.optimC = optim.Adamax(self.crt.parameters(), args.Clr,  betas=(0, 0.99), weight_decay=args.weight_decay)

    @staticmethod
    def real_fake_loss(crt_real, crt_fake):
        return -(crt_fake * (crt_real.detach() > crt_fake.detach()).float()).mean()

    @staticmethod
    def autoenc_loss(autoenc, crt, x, phase, alpha):
        x_fake   = autoenc(x, phase, alpha)
        loss_x   = utils.mismatch(x_fake, x, args.match_x_metric)
        crt_real = crt(x, x, phase, alpha)
        crt_fake = crt(x_fake, x, phase, alpha)
        loss_crt = Model.real_fake_loss(crt_real,crt_fake)
        return [args.autoenc_weight*loss_x, alpha*args.autoenc_weight**loss_crt], x_fake.detach()

    @staticmethod
    def mix_loss(mixcod, crt1, crt2, x1, x2, phase, alpha):
        x1_x2, x1_x1, z1_struct, z1_hat_struct = mixcod(x1, x2, phase, alpha)
        loss_x    = utils.mismatch(x1_x1, x1, args.match_x_metric)
        loss_z    = utils.mismatch(z1_struct, z1_hat_struct, args.match_z_metric)

        # compare cyclic x1 reconstruction with original x1
        crt1_real = crt1(x1, x1, phase, alpha)
        crt1_fake = crt1(x1_x1, x1, phase, alpha)
        loss_crt1 = Model.real_fake_loss(crt1_real,crt1_fake)
        # compare fake x1 translation with real x2
        crt2_real = crt2(x2, x2, phase, alpha)
        crt2_fake = crt2(x1_x2, x1_x2, phase, alpha)
        loss_crt2 = Model.real_fake_loss(crt2_real,crt2_fake)
        return [loss_x, loss_z, alpha*loss_crt1, alpha*loss_crt2], x1_x1.detach(), x1_x2.detach()

    @staticmethod
    def crt_loss_balanced(cr,cf):
        return cf - cr + torch.abs(cf + cr)

    @staticmethod
    def crt_loss_same_domain(crt, x, x_auto_fake, x_mix_fake, phase, alpha):
        crt_real        = crt(x, x, phase, alpha)
        crt_auto_fake   = crt(x_auto_fake, x, phase, alpha)
        crt_mix_fake    = crt(x_mix_fake, x, phase, alpha)
        cfa, cfm, cr    = crt_auto_fake.mean(), crt_mix_fake.mean(), crt_real.mean()
        return [args.autoenc_weight*Model.crt_loss_balanced(cr,cfa),
                args.autoenc_weight*Model.crt_loss_balanced(cr,cfm)], cr, cfa, cfm,

    @staticmethod
    def crt_loss_cross_domain(crt, cr, x_fake, phase, alpha):
        crt_fake    = crt(x_fake, x_fake, phase, alpha)
        cf          = crt_fake.mean()
        return [Model.crt_loss_balanced(cr,cf)], cf

    @staticmethod
    def autoenc_grad_norm(autoenc, x, phase, alpha):
        x_hat       = Variable(x, requires_grad=True)
        fake_x_hat  = autoenc(x_hat, phase, alpha)
        err_x_hat   = utils.mismatch(fake_x_hat, x_hat, args.match_x_metric)
        grad_x_hat  = grad(outputs=err_x_hat.sum(), inputs=x_hat, create_graph=True)[0]
        grad_norm   = grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1)
        return grad_norm

    @staticmethod
    def mixcod_grad_norm(mixcod, x1, x2, phase, alpha):
        x1_hat      = Variable(x1, requires_grad=True)
        # x2_hat      = Variable(x2, requires_grad=True)
        _,fake_x1_hat, z1_struct, z1_hat_struct = mixcod(x1_hat,x2,phase, alpha)
        err_x_hat   = utils.mismatch(fake_x1_hat, x1_hat, args.match_x_metric)
        err_z_hat   = utils.mismatch(z1_struct, z1_hat_struct, args.match_z_metric)
        loss = err_x_hat.sum() + err_z_hat.sum()
        grad_x_hat  = grad(outputs=loss, inputs=x1_hat, create_graph=True)[0]
        grad_norm   = grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1)
        return grad_norm

    @staticmethod
    def crt_grad_penalty(crt, x, fake_x, phase, alpha, grad_norm):
        eps = torch.rand(x.shape[1], 1, 1, 1).cuda()
        x_hat = eps * x.data + (1 - eps) * fake_x.data
        x_hat = Variable(x_hat, requires_grad=True)
        hat_predict = crt(x_hat, x, phase, alpha)
        grad_x_hat  = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        # Push the gradients of the interpolated samples towards 1
        loss_grad   = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - grad_norm) ** 2).mean()
        return [args.grad_weight*loss_grad]

    def update(self, batch, phase, alpha):
        ##### Train Encoder and Generator #########
        utils.requires_grad(self.enc, True)
        utils.requires_grad(self.gen, True)
        utils.requires_grad(self.crt, False)
        self.enc.zero_grad()
        self.gen.zero_grad()
        stats, losses = {}, []

        rgb,fir = batch
        # rgb = rgb.cuda(); fir = fir.cuda()

        ##### Autoencoders ###########
        loss_rgb, auto_fake_rgb = Model.autoenc_loss(self.autoenc.rgb, self.crt.rgb, rgb, phase, alpha)
        stats.update({'L1_auto_rgb':loss_rgb[0]})
        losses += loss_rgb

        loss_fir, auto_fake_fir = Model.autoenc_loss(self.autoenc.fir, self.crt.fir, fir, phase, alpha)
        stats.update({'L1_auto_fir':loss_fir[0]})
        losses += loss_fir

        ##### Mixcoders #############
        loss_rgb, mix_fake_rgb, mix_fake_rgb_fir = \
            Model.mix_loss(self.mixcod.rgb, self.crt.rgb, self.crt.fir, rgb, fir, phase, alpha)
        stats.update({'L1_mix_rgb':loss_rgb[0],'L2z_mix_rgb':loss_rgb[1]})
        losses  += loss_rgb

        loss_fir, mix_fake_fir, mix_fake_fir_rgb = \
            Model.mix_loss(self.mixcod.fir, self.crt.fir, self.crt.rgb, fir, rgb, phase, alpha)
        stats.update({'L1_mix_fir':loss_fir[0],'L2z_mix_fir':loss_fir[1]})
        losses  += loss_fir

        ##### Propagate gradients for encoders and generators
        loss     = sum(losses)
        loss.backward()
        self.optimE.step()
        self.optimG.step()

        ##### Train Critics ##########
        utils.requires_grad(self.enc, False)
        utils.requires_grad(self.gen, False)
        utils.requires_grad(self.crt, True)
        self.crt.zero_grad()

        losses = []

        # Same domain critics
        loss_crt_rgb, crt_real_rgb, crt_auto_fake_rgb, crt_mix_fake_rgb = \
            Model.crt_loss_same_domain(self.crt.rgb,rgb,auto_fake_rgb,mix_fake_rgb,phase,alpha)
        stats.update({'crt_real_rgb':crt_real_rgb,'crt_auto_fake_rgb':crt_auto_fake_rgb,'crt_mix_fake_rgb':crt_mix_fake_rgb})
        losses += loss_crt_rgb

        loss_crt_fir, crt_real_fir, crt_auto_fake_fir, crt_mix_fake_fir = \
            Model.crt_loss_same_domain(self.crt.fir,fir,auto_fake_fir,mix_fake_fir, phase,alpha)
        stats.update({'crt_real_fir':crt_real_fir,'crt_auto_fake_fir':crt_auto_fake_fir,'crt_mix_fake_fir':crt_mix_fake_fir})
        losses += loss_crt_fir

        # Cross domain critics
        loss_crt_mix_fir_rgb, crt_mix_fake_fir_rgb = \
            Model.crt_loss_cross_domain(self.crt.rgb,crt_real_rgb,mix_fake_fir_rgb,phase,alpha)
        stats.update({'crt_mix_fake_fir_rgb':crt_mix_fake_fir_rgb})
        losses += loss_crt_mix_fir_rgb

        loss_crt_mix_rgb_fir, crt_mix_fake_rgb_fir = \
            Model.crt_loss_cross_domain(self.crt.fir,crt_real_fir,mix_fake_rgb_fir,phase,alpha)
        stats.update({'crt_mix_fake_rgb_fir':crt_mix_fake_rgb_fir})
        losses += loss_crt_mix_rgb_fir

        ######### Gradient regularization #############

        # measure gradient of autoencoder
        gnorm_auto_rgb     = Model.autoenc_grad_norm(self.autoenc.rgb, rgb, phase, alpha).mean()
        # equalize graqient between real and fake samples
        loss_grad_auto_rgb = Model.crt_grad_penalty(self.crt.rgb, rgb, auto_fake_rgb, phase, alpha, args.grad_norm_fact*gnorm_auto_rgb)
        stats.update({'loss_grad_auto_rgb':loss_grad_auto_rgb[0]})
        losses += loss_grad_auto_rgb

        gnorm_auto_fir     = Model.autoenc_grad_norm(self.autoenc.fir, fir, phase, alpha).mean()
        # equalize graqient between real and fake samples
        loss_grad_auto_fir = Model.crt_grad_penalty(self.crt.fir, fir, auto_fake_fir, phase, alpha, args.grad_norm_fact*gnorm_auto_fir)
        stats.update({'loss_grad_auto_fir':loss_grad_auto_fir[0]})
        losses += loss_grad_auto_fir

        # measure gradient of mixcoder
        gnorm_mix_rgb     = Model.mixcod_grad_norm(self.mixcod.rgb, rgb, fir, phase, alpha).mean()
        loss_grad_mix_rgb = Model.crt_grad_penalty(self.crt.rgb, rgb, mix_fake_rgb, phase, alpha, args.grad_norm_fact*gnorm_mix_rgb)
        stats.update({'loss_grad_mix_rgb':loss_grad_mix_rgb[0]})
        losses += loss_grad_mix_rgb

        gnorm_mix_fir     = Model.mixcod_grad_norm(self.mixcod.fir, fir, rgb, phase, alpha).mean()
        loss_grad_mix_fir = Model.crt_grad_penalty(self.crt.fir, fir, mix_fake_fir, phase, alpha, args.grad_norm_fact*gnorm_mix_fir)
        stats.update({'loss_grad_mix_fir':loss_grad_mix_fir[0]})
        losses += loss_grad_mix_fir

        ##### Propagate gradients for critics
        loss = sum(losses)
        loss.backward()
        self.optimC.step()

        return stats

    def pbar_description(self,stats,batch,batch_count,sample_i,phase,alpha,res,epoch):
        return ('{0}; it: {1}; phase: {2}; batch: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; L1_mix_fir {7:.4f}; L1_mix_rgb {8:.4f}:').format(\
                batch_count+1, sample_i+1, phase, batch, alpha, res, epoch,
                stats['L1_mix_fir'], stats['L1_mix_rgb'])

    def dry_run(self, batch, phase, alpha):
        ''' dry run model on the batch '''
        self.enc.eval()
        self.gen.eval()
        utils.requires_grad(self.enc,False)
        utils.requires_grad(self.gen,False)

        rgb,fir    = batch
        batch_size = rgb.shape[0]

        rgb_fir, rgb_rgb, _, _ = self.mixcod.rgb(rgb,fir, phase,alpha)
        fir_rgb, fir_fir, _, _ = self.mixcod.fir(fir,rgb, phase,alpha)

        # join source and reconstructed images side by side
        out_ims    = torch.cat((rgb,rgb_rgb,rgb_fir,fir,fir_fir,fir_rgb), 1).view(6*batch_size,1,rgb.shape[-2],rgb.shape[-1])

        self.enc.train()
        self.gen.train()

        return out_ims

########### JUNK #################

        # gnorm_mix_rgb     = Model.autoenc_grad_norm(lambda rgb,phase,alpha: self.mixcod.rgb(rgb,phase,alpha)[1],
        #                                             rgb, phase, alpha).mean()
        # gnorm_mix_fir     = Model.autoenc_grad_norm(lambda fir,phase,alpha: self.mixcod.fir(fir,phase,alpha)[1],
        #                                             fir, phase, alpha).mean()

        # nchannels = z.shape[1]
        # rchannels = int(np.rint(nchannels*self.style_prop))

        # take z1 style and z2 struct
        # z2norm   = z2_style.norm(2, keepdim=True, dim=1)
        # z1_style = z2norm*z1_style/z1_style.norm(2, keepdim=True, dim=1)

        # # cross coder
        # gnorm_mix_rgb_fir  = Model.autoenc_grad_norm(lambda rgb,phase,alpha: self.mixcod.rgb(rgb,phase,alpha)[0],
        #                                             rgb, phase, alpha).mean()
        # loss_grad_mix_rgb_fir = Model.crt_grad_penalty(self.crt.fir, fir, mix_fake_rgb_fir, phase, alpha, gnorm_mix_rgb_fir)
        # stats.update({'loss_grad_mix_rgb_fir':loss_grad_mix_rgb_fir})
        #
        # gnorm_mix_fir_rgb  = Model.autoenc_grad_norm(lambda fir,phase,alpha: self.mixcod.fir(fir,phase,alpha)[0],
        #                                             fir, phase, alpha).mean()
        # loss_grad_mix_fir_rgb = Model.crt_grad_penalty(self.crt.rgb, fir, mix_fake_rgb_fir, phase, alpha, gnorm_mix_rgb_fir)
        # stats.update({'loss_grad_mix_rgb_fir':loss_grad_mix_rgb_fir})





