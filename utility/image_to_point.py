import torch

class toPoint(object):
    def __init__(self, height=32, width=32, normal_feature=3):
        self.height = height
        self.width = width
        self.normal_feature = normal_feature

    # def __call__(self, image):
    #     image_p = image.permute(1,2,0)
    #     image_pf = image_p.reshape(-1, 3)
    #     x = torch.arange(0, self.width)
    #     y = torch.arange(0, self.height)
    #     grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    #     grid_x = grid_x / self.width - 0.5
    #     grid_y = grid_y / self.height - 0.5
    #     grid_zero = torch.zeros([self.width * self.height, 1])
    #     out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1), grid_zero], dim=1)
    #     return out

    def __call__(self, image):
        image_p = image.permute(1,2,0)
        image_pf = image_p.reshape(-1, 3)
        x = torch.arange(0, self.width)
        y = torch.arange(0, self.height)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x = grid_x / self.width - 0.5
        grid_y = grid_y / self.height - 0.5
        out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1)], dim=1)
        return out


class toPointMnist(object):
    def __init__(self, height=28, width=28, channel=1, normal_feature=3):
        self.height = height
        self.width = width
        self.channel = channel
        self.normal_feature = normal_feature

    def __call__(self, image):
        image_p = image.permute(1,2,0)
        image_pf = image_p.reshape(-1, self.channel)
        x = torch.arange(0, self.width)
        y = torch.arange(0, self.height)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x = grid_x / self.width - 0.5
        grid_y = grid_y / self.height - 0.5

        if self.normal_feature == 3:
            out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1), image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1)], dim=1)
        elif self.normal_feature == 2:
            out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1), grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1)], dim=1)
        elif self.normal_feature == 1:
            out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1), image_pf], dim=1)
        else:
            out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1)], dim=1)

        # out = torch.cat([image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1), image_pf, grid_x.flatten().unsqueeze(1), grid_y.flatten().unsqueeze(1)], dim=1)
        return out