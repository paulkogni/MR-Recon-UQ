import torch
import torch.nn as nn
import numpy as np
import models.phirec.src.utils as utils

from models.phirec.src.torchlayers import Conv2D, Conv2DSequence, ReversibleSequence
from meddlr.ops import complex as cplx


seed = 66
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)


class DownConvolutionalBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        initializers,
        depth=3,
        padding=True,
        pool=True,
        reversible=False,
    ):
        super(DownConvolutionalBlock, self).__init__()

        if depth < 1:
            raise ValueError

        layers = []
        if pool:
            layers.append(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )

        if reversible:
            layers.append(ReversibleSequence(input_dim, output_dim, reversible_depth=3))
        else:
            layers.append(
                Conv2D(
                    input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
                )
            )

            if depth > 1:
                for i in range(depth - 1):
                    layers.append(
                        Conv2D(
                            output_dim,
                            output_dim,
                            kernel_size=3,
                            stride=1,
                            padding=int(padding),
                        )
                    )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpConvolutionalBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        initializers,
        padding,
        bilinear=True,
        reversible=False,
    ):
        super(UpConvolutionalBlock, self).__init__()
        self.bilinear = bilinear

        if self.bilinear:
            if reversible:
                self.upconv_layer = ReversibleSequence(
                    input_dim, output_dim, reversible_depth=2
                )
            else:
                self.upconv_layer = nn.Sequential(
                    Conv2D(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    Conv2D(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                )

        else:
            raise NotImplementedError

    def forward(self, x, bridge):
        if self.bilinear:
            x = nn.functional.interpolate(
                x, mode="bilinear", scale_factor=2, align_corners=True
            )
            x = self.upconv_layer(x)

        assert x.shape[3] == bridge.shape[3]
        assert x.shape[2] == bridge.shape[2]
        out = torch.cat([x, bridge], dim=1)

        return out


class SampleZBlock(nn.Module):
    """
    Performs 2 3X3 convolutions and a 1x1 convolution to mu and sigma which are used as parameters for a Gaussian
    for generating z
    """

    def __init__(self, input_dim, z_dim0=2, depth=2, reversible=False):
        super(SampleZBlock, self).__init__()
        self.input_dim = input_dim

        layers = []

        if reversible:
            layers.append(ReversibleSequence(input_dim, input_dim, reversible_depth=3))
        else:
            for i in range(depth):
                layers.append(Conv2D(input_dim, input_dim, kernel_size=3, padding=1))

        self.conv = nn.Sequential(*layers)

        self.mu_conv = nn.Sequential(nn.Conv2d(input_dim, z_dim0, kernel_size=1))
        self.sigma_conv = nn.Sequential(
            nn.Conv2d(input_dim, z_dim0, kernel_size=1), nn.Softplus()
        )

    def forward(self, pre_z):
        pre_z = self.conv(pre_z)
        mu = self.mu_conv(pre_z)
        sigma = self.sigma_conv(pre_z)

        z = mu + sigma * torch.randn_like(
            sigma, dtype=torch.float32
        )  

        return mu, sigma, z


class Posterior(nn.Module):
    """
    Posterior network of PHiSeg
    Returns a sample for each distribution of the latent level
    args:
        input_channels (int): number of input_channels, one for greyscale
        is_posterior (boolean): if True, mask is concatenated to inpout of encoder -> gets conditional VAE
        num_classes (int): number of classes to classify for
        num_filters (list or 1D array): list of the number of filters to apply
    """

    def __init__(
        self,
        input_channels,
        num_classes,
        num_filters,
        initializers,
        padding=True,
        is_posterior=True,
        reversible=False,
    ):
        super(Posterior, self).__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters

        self.latent_levels = 5
        self.resolution_levels = 7

        self.lvl_diff = self.resolution_levels - self.latent_levels

        self.padding = padding
        self.activation_maps = []

        # increase input by two for conditional VAE
        if is_posterior:
            self.input_channels += 2  # 1

        self.contracting_path = nn.ModuleList()

        for i in range(self.resolution_levels):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            pool = False if i == 0 else True

            self.contracting_path.append(
                DownConvolutionalBlock(
                    input,
                    output,
                    initializers,
                    depth=3,
                    padding=padding,
                    pool=pool,
                    reversible=reversible,
                )
            )

        self.upsampling_path = nn.ModuleList()

        for i in reversed(range(self.latent_levels)):
            input = 2
            output = self.num_filters[0] * 2
            self.upsampling_path.append(
                UpConvolutionalBlock(
                    input, output, initializers, padding, reversible=reversible
                )
            )

        self.sample_z_path = nn.ModuleList()

        for i in reversed(range(self.latent_levels)):
            input = 2 * self.num_filters[0] + self.num_filters[i + self.lvl_diff]

            if i == self.latent_levels - 1:
                input = self.num_filters[i + self.lvl_diff]
                self.sample_z_path.append(
                    SampleZBlock(input, depth=2, reversible=reversible)
                )
            else:
                self.sample_z_path.append(
                    SampleZBlock(input, depth=2, reversible=reversible)
                )

    def forward(self, patch, segm=None, training_prior=False, z_list=None):
        if segm is not None:
            patch = torch.cat([patch, segm], dim=1)

        blocks = []
        z = [None] * self.latent_levels
        mu = [None] * self.latent_levels
        sigma = [None] * self.latent_levels

        x = patch
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        pre_conv = x

        for i, sample_z in enumerate(self.sample_z_path):
            if i != 0:
                pre_conv = self.upsampling_path[i - 1](z[-i], blocks[-i])
            mu[-i - 1], sigma[-i - 1], z[-i - 1] = self.sample_z_path[i](pre_conv)
            if training_prior:
                z[-i - 1] = z_list[-i - 1]

        del blocks

        return z, mu, sigma


def increase_resolution(times, input_dim, output_dim):
    """Increase the resolution by n time for the beginning of the likelihood path"""
    module_list = []
    for i in range(times):
        module_list.append(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True)
        )
        if i != 0:
            input_dim = output_dim
        module_list.append(
            Conv2DSequence(input_dim=input_dim, output_dim=output_dim, depth=1)
        )

    return nn.Sequential(*module_list)


class Likelihood(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        num_filters,
        latent_levels=5,
        resolution_levels=7,
        image_size=(1, 128, 128),
        reversible=False,
        initializers=None,
        apply_last_layer=True,
        padding=True,
    ):
        super(Likelihood, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.lvl_diff = resolution_levels - latent_levels

        self.image_size = image_size
        self.reversible = reversible

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        # LIKELIHOOD
        self.likelihood_ups_path = nn.ModuleList()
        self.likelihood_post_ups_path = nn.ModuleList()

        # path for upsampling
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i]
            output = self.num_filters[i]
            if reversible:
                self.likelihood_ups_path.append(
                    ReversibleSequence(
                        input_dim=2, output_dim=input, reversible_depth=2
                    )
                )
            else:
                self.likelihood_ups_path.append(
                    Conv2DSequence(input_dim=2, output_dim=input, depth=2)
                )

            self.likelihood_post_ups_path.append(
                increase_resolution(
                    times=self.lvl_diff, input_dim=input, output_dim=input
                )
            )

        # path after concatenation
        self.likelihood_post_c_path = nn.ModuleList()
        for i in range(latent_levels - 1):
            input = self.num_filters[i] + self.num_filters[i + 1 + self.lvl_diff]
            output = self.num_filters[i + self.lvl_diff]

            if reversible:
                self.likelihood_post_c_path.append(
                    ReversibleSequence(
                        input_dim=input, output_dim=output, reversible_depth=2
                    )
                )
            else:
                self.likelihood_post_c_path.append(
                    Conv2DSequence(input_dim=input, output_dim=output, depth=2)
                )

        self.s_layer = nn.ModuleList()
        output = self.num_classes
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i + self.lvl_diff]
            self.s_layer.append(
                Conv2DSequence(
                    input_dim=input,
                    output_dim=output,
                    depth=1,
                    kernel=1,
                    activation=torch.nn.Identity,
                    norm=torch.nn.Identity,
                )
            )

    def forward(self, z):
        """Likelihood network which takes list of latent variables z with dimension latent_levels"""
        s = [None] * self.latent_levels
        post_z = [None] * self.latent_levels
        post_c = [None] * self.latent_levels

        # start from the downmost layer and the last filter
        for i in range(self.latent_levels):
            assert z[-i - 1].shape[1] == 2
            assert z[-i - 1].shape[2] == self.image_size[1] * 2 ** (
                -self.resolution_levels + 1 + i
            )
            post_z[-i - 1] = self.likelihood_ups_path[i](z[-i - 1])

            post_z[-i - 1] = self.likelihood_post_ups_path[i](post_z[-i - 1])
            assert post_z[-i - 1].shape[2] == self.image_size[1] * 2 ** (
                -self.latent_levels + i + 1
            )
            assert (
                post_z[-i - 1].shape[1] == self.num_filters[-i - 1 - self.lvl_diff]
            ), "{} != {}".format(post_z[-i - 1].shape[1], self.num_filters[-i - 1])

        post_c[self.latent_levels - 1] = post_z[self.latent_levels - 1]

        for i in reversed(range(self.latent_levels - 1)):
            ups_below = nn.functional.interpolate(
                post_c[i + 1], mode="bilinear", scale_factor=2, align_corners=True
            )

            assert post_z[i].shape[3] == ups_below.shape[3]
            assert post_z[i].shape[2] == ups_below.shape[2]

            # Reminder: Pytorch standard is NCHW, TF NHWC
            concat = torch.cat([post_z[i], ups_below], dim=1)

            post_c[i] = self.likelihood_post_c_path[i](concat)

        for i, block in enumerate(self.s_layer):
            s_in = block(post_c[-i - 1])  # no activation in the last layer
            s[-i - 1] = torch.nn.functional.interpolate(
                s_in, size=[self.image_size[1], self.image_size[2]], mode="bilinear"
            )  # mode='nearest')

        return s


class PHIRec(nn.Module):
    """
    Implementation of PHiSeg by Baumgartner

    """

    def __init__(
        self,
        input_channels,
        num_classes,
        num_filters,
        latent_levels=5,
        latent_dim=2,
        initializers=None,
        no_convs_fcomb=4,
        beta=10.0,
        image_size=None,
        reversible=False,
        apply_last_layer=True,
        exponential_weighting=True,
        padding=True,
        loss_fn=torch.nn.MSELoss(reduction="none"),
    ):
        super(PHIRec, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = latent_levels
        self.image_size = image_size

        self.loss_tot = 0

        self.loss_dict = {}
        self.kl_divergence_loss_weight = 1.0

        self.beta = 1.0

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.exponential_weighting = exponential_weighting
        self.exponential_weight = 4
        self.residual_multinoulli_loss_weight = 1.0

        self.kl_divergence_loss = 0.0
        self.reconstruction_loss = 0.0

        self.posterior = Posterior(
            input_channels,
            num_classes,
            num_filters,
            initializers=None,
            padding=True,
            reversible=reversible,
        )
        self.likelihood = Likelihood(
            input_channels,
            num_classes,
            num_filters,
            initializers=None,
            apply_last_layer=True,
            padding=True,
            image_size=self.image_size,
            reversible=reversible,
        )
        self.prior = Posterior(
            input_channels,
            num_classes,
            num_filters,
            initializers=None,
            padding=True,
            is_posterior=False,
            reversible=reversible,
        )
        self.s_out_list = [None] * self.latent_levels
        self.s_out_list_with_softmax = [None] * self.latent_levels

        self.loss_fn = loss_fn

    def sample_posterior(self):
        z_sample = [None] * self.latent_levels
        mu = self.posterior_mu
        sigma = self.posterior_sigma

        for i, _ in enumerate(z_sample):
            z_sample[i] = mu[i] + sigma[i] * torch.randn_like(
                sigma[i]
            )  

        return z_sample

    def sample_prior(self):
        z_sample = [None] * self.latent_levels
        mu = self.prior_mu
        sigma = self.prior_sigma

        for i, _ in enumerate(z_sample):
            z_sample[i] = mu[i] + sigma[i] * torch.randn_like(
                sigma[i]
            )  

        return z_sample

    def sample(self, testing=True):
        if testing:
            sample, _ = self.reconstruct(self.sample_prior())
        else:
            raise NotImplementedError
        return sample

    def reconstruct(self, z_posterior):
        layer_recon = self.likelihood(z_posterior)
        return self.accumulate_output(layer_recon), layer_recon

    def forward(self, patch, mask, training=True):
        self.patch = patch
        if training:
            (
                self.posterior_latent_space,
                self.posterior_mu,
                self.posterior_sigma,
            ) = self.posterior(patch, mask)
            self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior(
                patch, training_prior=True, z_list=self.posterior_latent_space
            )
            self.s_out_list = self.likelihood(self.posterior_latent_space)
        else:
            (
                self.posterior_latent_space,
                self.posterior_mu,
                self.posterior_sigma,
            ) = self.posterior(patch, mask)
            self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior(
                patch, training_prior=False
            )
            self.s_out_list = self.likelihood(self.prior_latent_space)

        return self.s_out_list

    def accumulate_output(self, output_list):
        s_accum = output_list[-1].clone()
        for i in range(len(output_list) - 1):
            s_accum += output_list[i]
        return s_accum

    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):
        sigma0_fs = torch.mul(
            torch.flatten(sigma0, start_dim=1), torch.flatten(sigma0, start_dim=1)
        )
        sigma1_fs = torch.mul(
            torch.flatten(sigma1, start_dim=1), torch.flatten(sigma1, start_dim=1)
        )

        logsigma0_fs = torch.log(sigma0_fs + 1e-10)
        logsigma1_fs = torch.log(sigma1_fs + 1e-10)

        mu0_f = torch.flatten(mu0, start_dim=1)
        mu1_f = torch.flatten(mu1, start_dim=1)

        return torch.mean(
            0.5
            * torch.sum(
                torch.div(
                    sigma0_fs + torch.mul((mu1_f - mu0_f), (mu1_f - mu0_f)),
                    sigma1_fs + 1e-10,
                )
                + logsigma1_fs
                - logsigma0_fs
                - 1,
                dim=1,
            )
        )

    def calculate_hierarchical_KL_div_loss(self):
        prior_sigma_list = self.prior_sigma
        prior_mu_list = self.prior_mu
        posterior_sigma_list = self.posterior_sigma
        posterior_mu_list = self.posterior_mu

        kl_loss = 0.0

        if self.exponential_weighting:
            level_weights = [
                self.exponential_weight**i for i in list(range(self.latent_levels))
            ]
        else:
            level_weights = [1] * self.exp_config.latent_levels

        for ii, mu_i, sigma_i in zip(
            reversed(range(self.latent_levels)),
            reversed(posterior_mu_list),
            reversed(posterior_sigma_list),
        ):
            self.loss_dict["KL_divergence_loss_lvl%d" % ii] = level_weights[
                ii
            ] * self.KL_two_gauss_with_diag_cov(
                mu_i, sigma_i, prior_mu_list[ii], prior_sigma_list[ii]
            )

            kl_loss += (
                self.kl_divergence_loss_weight
                * self.loss_dict["KL_divergence_loss_lvl%d" % ii]
            )

        return kl_loss

    def multinoulli_loss(self, reconstruction, target):
        criterion = self.loss_fn

        batch_size = reconstruction.shape[0]

        recon_flat = reconstruction.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        return torch.mean(
            torch.sum(criterion(target=target_flat, input=recon_flat), dim=1)
        )

    def residual_multinoulli_loss(self, reconstruction, target):
        self.s_accumulated = [None] * self.latent_levels

        criterion = self.multinoulli_loss
        recon_loss = 0.0

        for ii, s_ii in zip(
            reversed(range(self.latent_levels)), reversed(reconstruction)
        ):
            if ii == self.latent_levels - 1:
                self.s_accumulated[ii] = s_ii

                self.loss_dict["residual_multinoulli_loss_lvl%d" % ii] = criterion(
                    self.s_accumulated[ii] + self.patch, target
                )

            else:
                self.s_accumulated[ii] = self.s_accumulated[ii + 1] + s_ii
                self.loss_dict["residual_multinoulli_loss_lvl%d" % ii] = criterion(
                    self.s_accumulated[ii] + self.patch, target
                )

            recon_loss += (
                self.residual_multinoulli_loss_weight
                * self.loss_dict["residual_multinoulli_loss_lvl%d" % ii]
            )
        return recon_loss

    def loss(self, segm):
        self.loss_tot = 0.0
        self.kl_divergence_loss = 0.0
        self.reconstruction_loss = 0.0

        self.kl_divergence_loss = self.calculate_hierarchical_KL_div_loss()
        self.reconstruction_loss = self.residual_multinoulli_loss(
            reconstruction=self.s_out_list, target=segm
        )

        self.loss_tot = torch.add(
            self.reconstruction_loss,
            self.kl_divergence_loss_weight * self.kl_divergence_loss,
        )

        return self.loss_tot

    def make_prediction(self, img):
        """Performs a prediction to obtain a reconstruction

        Args:
            img (torch.Tensor): The undersampled image to reconstruct with shape (n_batch, n_channel, width, height)

        Returns:
            torch.Tensor: The reconstruction from multiple samples with same shape as input
        """
        if torch.cuda.is_available():
            self.cuda()
            img = img.cuda()
        samples = []
        for i in range(20):
            prior_latent_space, _, _ = self.prior(img, training_prior=False)
            s_out_list = self.likelihood(prior_latent_space)
            accumulated = self.accumulate_output(s_out_list)
            samples.append(accumulated)
        return torch.stack(samples)


def compute_train_loss_and_train(train_loader, model, optimizer, device, epoch):
    """
    computes the losses for every batch; so basically the epoch loss
    """
    model.train()

    running_loss = 0.0
    kl_running_loss = 0.0
    recon_running_loss = 0.0

    for x, y, _, _ in train_loader:
        if device:
            x = x.to(device)
            y = y.to(device)

        # forward pass
        outputs = model(x, y)

        # compute loss
        loss = model.loss(y)

        # save losses for later
        loss_tot = model.loss_tot
        kl_loss = model.kl_divergence_loss
        rec_loss = model.reconstruction_loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track running loss
        running_loss += loss_tot * train_loader.batch_size
        kl_running_loss += kl_loss * train_loader.batch_size
        recon_running_loss += rec_loss * train_loader.batch_size
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_kl_loss = kl_running_loss / len(train_loader.dataset)
    epoch_recon_loss = recon_running_loss / len(train_loader.dataset)

    return epoch_loss, epoch_kl_loss, epoch_recon_loss


def compute_eval_loss(test_loader, model, device, epoch):
    """
    computes the evaluation epoch loss on the evaluation set
    """
    model.eval()

    running_loss = 0.0
    kl_running_loss = 0.0
    recon_running_loss = 0.0
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            if device:
                x = x.to(device)
                y = y.to(device)

            # forward pass
            outputs = model(x, y)

            # compute loss
            loss = model.loss(y)

            # save losses for later
            loss_tot = model.loss_tot
            kl_loss = model.kl_divergence_loss
            rec_loss = model.reconstruction_loss

            # backward pass

            # track running loss
            running_loss += loss_tot * test_loader.batch_size
            kl_running_loss += kl_loss * test_loader.batch_size
            recon_running_loss += rec_loss * test_loader.batch_size
    torch.cuda.empty_cache()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_kl_loss = kl_running_loss / len(test_loader.dataset)
    epoch_recon_loss = recon_running_loss / len(test_loader.dataset)

    return epoch_loss, epoch_kl_loss, epoch_recon_loss


def train_model(
    model,
    train_loader,
    eval_loader,
    optim,
    epochs=1,
    save_model=None,
    save_path=None,
    continue_training_path=None,
    eval_metric=None,
    device="cuda:0",
):
    """
    Trains and evaluates the model. Additionally, training stats can be tracked and best performing models (according to evaluation loss and GED) can be saveed
    params:
        model: torch module. The PHiSeg model to train
        train_loader: torch data loader (for training data)
        eval_loader: torch data loader (for evaluation)
        optim: torch optimizer
        epochs: int. The number of epochs to train
        save_model: boolean (optional). In case you want to save you best performing models during training
        save_path: string (optional but must be defined if save_model is True). The location where you want to save the model
        continue_training_path: string (optional): The path to the model with state dict and optimizer state for continuing training
        eval_ged: boolean (optional): if set true, the GED between samples and ground truth labels is being computed. Only works with multiple labels for one data point
    """

    end_epoch = 0
    # use_gpu = torch.cuda.is_available()
    print("Using GPU:", device)
    if continue_training_path:
        checkpoint = torch.load(continue_training_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if device:
            model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        end_epoch = checkpoint["epoch"]
    if device:
        model.to(device)

    # define current best losses
    best_total_eval_loss = np.inf
    best_ssim = -np.inf

    for epoch in range(end_epoch, epochs):
        print("Epoch", epoch)
        # train the model and compute train loss
        (
            train_running_loss,
            train_kl_loss,
            train_recon_loss,
        ) = compute_train_loss_and_train(
            train_loader, model, optim, device, epoch=epoch
        )

        # compute evaluation loss
        eval_running_loss, eval_kl_loss, eval_recon_loss = compute_eval_loss(
            eval_loader, model, device, epoch
        )

        # compute GED between samples and GTs on evaluation set
        if eval_metric:
            if epoch % 50 == 0:  # compute only every 50 epochs
                psnr, ssim, _ = utils.eval_ssim_psnr_ncc(
                    model, eval_loader, model_type="phiseg", n_samples=10, device=device
                )
                print("psnr:", psnr)
                print("ssim:", ssim)

        if save_model == True:
            if eval_running_loss < best_total_eval_loss:
                best_total_eval_loss = eval_running_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": train_running_loss,
                    },
                    f"{save_path}phiseg_best_eval_epoch{epoch}.pth",
                )
                print("saving best eval model")
            if eval_metric:
                if epoch % 50 == 0:
                    if ssim > best_ssim:
                        best_ssim = ssim
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optim.state_dict(),
                                "loss": train_running_loss,
                            },
                            f"{save_path}phiseg_best_ged_epoch{epoch}.pth",
                        )
                        print("saving best GED model")

        print("training loss:", train_running_loss)
        print("evaluation loss:", eval_running_loss)

    return
