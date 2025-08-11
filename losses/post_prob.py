import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device=None):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.use_bg = use_background
        self.device = device or torch.device("cuda")

        self.cood = torch.arange(0, c_size, step=stride, dtype=torch.float32, device=self.device) + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, points, st_sizes):
        # Ensure st_sizes is a tensor and move it to the correct device
        st_sizes = st_sizes.to(self.device)
        
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0).to(self.device)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for i, dis in enumerate(dis_list):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        bg_dis = (st_sizes[i] * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                    prob_list.append(prob)
                else:
                    prob_list.append(None)
        else:
            prob_list = [None] * len(points)
        return prob_list