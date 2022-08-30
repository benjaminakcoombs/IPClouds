import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class TimelapseDatasetMask(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, path, stage="train", image_size=(480, 640), world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.base_path = path + "_" + stage
#        self.base_path = os.path.join(path, "fullRender", "final", "fullRender") + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = 0
        self.z_far = 1
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        mask_paths = sorted(glob.glob(os.path.join(dir_path, "mask", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        count = 0
        for rgb_path, pose_path, mask_path in zip(rgb_paths, pose_paths, mask_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = imageio.imread(mask_path)
            #mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
#            print("Mask is:")
#            print(mask)


            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose[:3, 3] = pose[:3, 3] / 350
            pose = pose @ self._coord_trans

            bbox = []
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    if mask[i, j] > 200:
                        bbox.append([j, i, True, count])
                    else:
                        bbox.append([j, i, False, count])
            bbox = torch.tensor(bbox, dtype=torch.long)

            all_imgs.append(img_tensor)
            mask_tensor = self.mask_to_tensor(mask)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)
            count = count + 1

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)
#        print("******************************",all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)
#        print("BBOxes are:",all_bboxes)
        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
