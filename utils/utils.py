from torch.utils.data import Dataset
import os
from PIL import Image
from utils.data_prep import discover_images, load_rgb, read_manifest
from torchvision import transforms


class ImageFolderDataset(Dataset):
    """Images from a manifest (.txt, one path per line), a list of paths, or a
    directory (searched recursively, case-insensitive extensions).

    All images are loaded fully and converted to RGB (RGBA composited on white).
    Originals are never modified.
    """

    def __init__(self, source, transform=None, max_retries=5):
        super(ImageFolderDataset, self).__init__()
        self.transform = transform
        self.max_retries = max_retries
        if isinstance(source, (list, tuple)):
            self.files = [str(p) for p in source]
        elif os.path.isfile(str(source)):
            self.files = read_manifest(source)
        else:
            self.files, _ = discover_images(source)
        if not self.files:
            raise ValueError(f'No images found for source: {source}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Manifests are pre-validated, so failures should be rare (e.g. a file
        # deleted after preparation). Never fail silently: warn, then try the
        # next image so one bad file cannot kill a long run.
        for attempt in range(self.max_retries):
            image_path = self.files[(idx + attempt) % len(self.files)]
            try:
                image = load_rgb(image_path)
                break
            except Exception as exc:  # noqa: BLE001
                print(f'[ImageFolderDataset] WARNING: could not load {image_path}: {exc!r}',
                      flush=True)
        else:
            raise RuntimeError(f'{self.max_retries} consecutive unreadable images from index {idx}')

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        

def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std