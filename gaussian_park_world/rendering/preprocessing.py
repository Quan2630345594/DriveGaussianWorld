"""Reproduce saved model input images for offline rendering metrics."""

import numpy as np


def load_images(paths, size=518, input_intrinsics=None, horizontal_fov_deg=None):
    import torch
    from PIL import Image
    tensors = []
    processed_intrinsics = []
    for index, path in enumerate(paths):
        image = Image.open(path).convert('RGB')
        w, h = image.size
        intrinsic = None if input_intrinsics is None else np.array(input_intrinsics[index], dtype=float)
        if horizontal_fov_deg is not None:
            import cv2
            target = intrinsic.copy()
            target[0, 0] = w / (2 * np.tan(np.deg2rad(horizontal_fov_deg / 2)))
            target[1, 1] *= target[0, 0] / intrinsic[0, 0]
            image = Image.fromarray(cv2.warpPerspective(
                np.array(image), target @ np.linalg.inv(intrinsic), (w, h),
                flags=cv2.INTER_LINEAR))
            intrinsic = target
        new_h = round(h * (size / w) / 14) * 14
        image = image.resize((size, new_h), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.
        if new_h > size:
            top = (new_h - size) // 2
            tensor = tensor[:, top:top + size]
        if intrinsic is not None:
            transform = np.diag([size / w, new_h / h, 1.])
            if new_h > size:
                transform[1, 2] = -top
            processed_intrinsics.append(transform @ intrinsic)
        tensors.append(tensor)
    return torch.stack(tensors), np.array(processed_intrinsics)
