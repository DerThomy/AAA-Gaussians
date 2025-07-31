import torch 

# 3D Mip filter sample rate calculation from:
# https://github.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py
def mip_filter_3d(xyz, cameras):
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
    
    # we should use the focal length of the highest resolution camera
    focal_length = 0.
    for camera in cameras:

        # transform points to camera space
        R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
        xyz_cam = xyz @ R + T[None, :]
        
        xyz_to_cam = torch.norm(xyz_cam, dim=1)
        
        # project to screen space
        valid_depth = xyz_cam[:, 2] > 0.2
        
        
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * camera.focal_x + camera.image_width / 2.0
        y = y / z * camera.focal_y + camera.image_height / 2.0
        
        # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
        
        # use similar tangent space filtering as in the paper
        in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
        
    
        valid = torch.logical_and(valid_depth, in_screen)
        
        # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
        distance[valid] = torch.min(distance[valid], z[valid])
        valid_points = torch.logical_or(valid_points, valid)
        if focal_length < camera.focal_x:
            focal_length = camera.focal_x
    
    distance[~valid_points] = 0
    
    filter_3D = distance / focal_length * (0.3 ** 0.5)
    return filter_3D[..., None]