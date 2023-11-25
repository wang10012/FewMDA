import torch
from test_diffusion.models.improved_ddpm.script_util import i_DDPM
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from test_diffusion.diffusion_utils import get_beta_schedule,denoising_step
from tqdm import tqdm
import torchvision
import torchvision.utils as tvu
import os

# utils
def show_image(x):
    x = (x + 1) * 0.5
    tensor_cpu = x.cpu()
    image = tensor_cpu.squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.show()

def luma_transform(input):
    assert input.shape[0] == 3, "Input must have 3 channels"
    R, G, B = input[0, :, :], input[1, :, :], input[2, :, :]
    L = R * 0.299 + G * 0.587 + B * 0.114
    L = L.unsqueeze(0).expand(3, -1, -1)
    return L

# dataset class
class Cub2011Painting(torchvision.datasets.ImageFolder):

    def __getitem__(self, idx):
        path, _ = self.samples[idx]

        img =  Image.open(path)

        img = img.resize((512,512), Image.ANTIALIAS)
        img = np.array(img) / 255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2,0,1)
        img = (img - 0.5) * 2

        img = luma_transform(img)

        return {
            "image": img,
            "filename": path,
        }

# load model
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("device:",device)

model = i_DDPM("IMAGENET") # imagenet
learn_sigma = True
ckpt = torch.load("./test_diffusion/512x512_diffusion.pt")
model.load_state_dict(ckpt)
model.to(device)
model.eval()
print("Improved diffusion Model loaded.")

# define variables
n_inv_step = 40
n_test_step = 40
t_0  = 601

class config: 
    class diffusion:
        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        num_diffusion_timesteps = 1000

betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )

alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

betas = torch.from_numpy(betas).float().to(device)

logvar = np.log(np.maximum(posterior_variance, 1e-20))

# construct dataloader
dataset = Cub2011Painting('./data/CUB-Few-Painting', transform=None)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# loop
for step,batch in enumerate(loader):
    print("*******************************",step,"*******************************")
    img = batch['image']
    path = batch['filename']
    x0 = img.to(device)

    with torch.no_grad():
        # inverse process
        seq_inv = np.linspace(0, 1, n_inv_step) * t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        x = x0.clone()
        with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
            for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                t = (torch.ones(1) * i).to(device)
                t_prev = (torch.ones(1) * j).to(device)
                x = denoising_step(
                    x, t=t, t_next=t_prev, models=model, 
                    logvars=logvar, sampling_type='ddim', 
                    b=betas, eta=0, learn_sigma=learn_sigma, 
                    ratio=0,
                    )

                progress_bar.update(1)
            x_lat = x.clone()
        
        # reverse process
        print(f"Sampling type: {'ddim'.upper()} with eta {0.0}, "
              f" Steps: {n_test_step}/{t_0}")
        seq_test = np.linspace(0, 1, n_test_step) * t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        n_iter = 5
        for it in range(n_iter):
            x = x_lat.clone()
            with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(1) * i).to(device)
                    t_next = (torch.ones(1) * j).to(device)
                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                            logvars=logvar,
                                            sampling_type='ddim',
                                            b=betas,
                                            eta=0.0,
                                            learn_sigma=learn_sigma,
                                            ratio=0,
                                            )
                    
                    progress_bar.update(1)

        tvu.save_image((x + 1) * 0.5, os.path.join("./data/CUB-DDIM-Painting",
                                                   f'{step+1}.jpg'))
        print('success saved!')