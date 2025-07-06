import torch
from lpipsPyTorch.modules.lpips import LPIPS

x = torch.rand(3, 64, 64).unsqueeze(0).cuda()
y = torch.rand(3, 64, 64).unsqueeze(0).cuda()

model = LPIPS(net_type='vgg').to('cuda').eval()

scripted = torch.jit.trace(model, (x, y))
scripted.save('lpips_vgg.pt')