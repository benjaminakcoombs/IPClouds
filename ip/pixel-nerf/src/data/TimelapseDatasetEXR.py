import os
import torch
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class TimelapseDatasetEXR(torch.utils.data.Dataset):
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
        #self.base_path = os.path.join(path, "fullRender", "final", "fullRender") + "_" + stage
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        print(self.base_path)
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

        self.z_near = 0.1
        self.z_far = 1
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)



    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        gt_paths = sorted(glob.glob(os.path.join(dir_path, "ground_truth", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())
        def project_points(world_points, Tcw):
            N = world_points.shape[0]
            world_points_h = np.concatenate([world_points, np.ones((N,1))],axis=1)
            cam_points = Tcw @ world_points_h.T 
            cam_points = cam_points[:3,:].T
            img_points = np.zeros((N,3))
            img_points[:,0] = focal * cam_points[:,0] / cam_points[:,2] + cx
            img_points[:,1] = focal * cam_points[:,1] / cam_points[:,2] + cy
            img_points[:,2] = cam_points[:,2] - Tcw[2,3]
            return img_points

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path, gt_path in zip(rgb_paths, pose_paths, gt_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            coord_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[:,:,::-1]
            coords = np.reshape(coord_img, [640*480,3])
            coords[:,1] = -coords[:,1]

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )            
            pose[:3, 3] = pose[:3, 3] / 10000
            pose = pose @ self._coord_trans
            Tcw = np.linalg.inv(pose)
            img_coords = project_points(coords, Tcw)
            mask = np.reshape(img_coords[:, 1:],[480,640,2])[:,:,1]
            mask = mask / -10000
            mask_tensor = self.mask_to_tensor(mask)

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

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

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": [],
            "poses": all_poses,
        }
        
        return result


    