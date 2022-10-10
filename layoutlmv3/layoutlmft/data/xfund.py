import os
import json

import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from typing import List, Union
from transformers.utils import is_torch_available
from transformers.utils.generic import _is_torch

def is_torch_tensor(obj):
    return _is_torch(obj) if is_torch_available() else False

def _ensure_format_supported(image):
    if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
        raise ValueError(
            f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
            "`torch.Tensor` are."
        )

def to_pil_image(image, rescale=None):
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.
    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
            The image to convert to the PIL Image format.
        rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
            default to `True` if the image type is a floating type, `False` otherwise.
    """
    _ensure_format_supported(image)

    if is_torch_tensor(image):
        image = image.numpy()

    if isinstance(image, np.ndarray):
        if rescale is None:
            # rescale default to the array being of floating type.
            rescale = isinstance(image.flat[0], np.floating)
        # If the channel as been moved to first dim, we put it back at the end.
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        if rescale:
            image = image * 255
        image = image.astype(np.uint8)
        return PIL.Image.fromarray(image)
    return image

def convert_rgb(image):
    """
    Converts `PIL.Image.Image` to RGB format.
    Args:
        image (`PIL.Image.Image`):
            The image to convert.
    """
    _ensure_format_supported(image)
    if not isinstance(image, PIL.Image.Image):
        return image

    return image.convert("RGB")

def rescale(image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
    """
    Rescale a numpy image by scale amount
    """
    _ensure_format_supported(image)
    return image * scale

def to_numpy_array(image, rescale=None, channel_first=True):
    """
    Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
    dimension.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to convert to a NumPy array.
        rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
            default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
        channel_first (`bool`, *optional*, defaults to `True`):
            Whether or not to permute the dimensions of the image to put the channel dimension first.
    """
    _ensure_format_supported(image)

    if isinstance(image, PIL.Image.Image):
        image = np.array(image)

    if is_torch_tensor(image):
        image = image.numpy()

    rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

    if rescale:
        image = rescale(image.astype(np.float32), 1 / 255.0)

    if channel_first and image.ndim == 3:
        image = image.transpose(2, 0, 1)

    return image

def expand_dims(image):
    """
    Expands 2-dimensional `image` to 3 dimensions.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to expand.
    """
    _ensure_format_supported(image)

    # Do nothing if PIL image
    if isinstance(image, PIL.Image.Image):
        return image

    if is_torch_tensor(image):
        image = image.unsqueeze(0)
    else:
        image = np.expand_dims(image, axis=0)
    return image

def normalize(image, mean, std, rescale=False):
    """
    Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
    if it's a PIL Image.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to normalize.
        mean (`List[float]` or `np.ndarray` or `torch.Tensor`):
            The mean (per channel) to use for normalization.
        std (`List[float]` or `np.ndarray` or `torch.Tensor`):
            The standard deviation (per channel) to use for normalization.
        rescale (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
            happen automatically.
    """
    _ensure_format_supported(image)

    if isinstance(image, PIL.Image.Image):
        image = to_numpy_array(image, rescale=True)
    # If the input image is a PIL image, it automatically gets rescaled. If it's another
    # type it may need rescaling.
    elif rescale:
        if isinstance(image, np.ndarray):
            image = rescale(image.astype(np.float32), 1 / 255.0)
        elif is_torch_tensor(image):
            image = rescale(image.float(), 1 / 255.0)

    if isinstance(image, np.ndarray):
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean).astype(image.dtype)
        if not isinstance(std, np.ndarray):
            std = np.array(std).astype(image.dtype)
    elif is_torch_tensor(image):
        import torch

        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)

    if image.ndim == 3 and image.shape[0] in [1, 3]:
        return (image - mean[:, None, None]) / std[:, None, None]
    else:
        return (image - mean) / std

def resize(image, size, resample=PIL.Image.BILINEAR, default_to_square=True, max_size=None):
    """
    Resizes `image`. Enforces conversion of input to PIL.Image.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be
            matched to this.
            If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
            `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to
            this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            The filter to user for resampling.
        default_to_square (`bool`, *optional*, defaults to `True`):
            How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a
            square (`size`,`size`). If set to `False`, will replicate
            [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
            with support for resizing only the smallest edge and providing an optional `max_size`.
        max_size (`int`, *optional*, defaults to `None`):
            The maximum allowed for the longer edge of the resized image: if the longer edge of the image is
            greater than `max_size` after being resized according to `size`, then the image is resized again so
            that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller
            edge may be shorter than `size`. Only used if `default_to_square` is `False`.
    Returns:
        image: A resized `PIL.Image.Image`.
    """
    _ensure_format_supported(image)

    if not isinstance(image, PIL.Image.Image):
        image = to_pil_image(image)

    if isinstance(size, list):
        size = tuple(size)

    if isinstance(size, int) or len(size) == 1:
        if default_to_square:
            size = (size, size) if isinstance(size, int) else (size[0], size[0])
        else:
            width, height = image.size
            # specified size only for the smallest edge
            short, long = (width, height) if width <= height else (height, width)
            requested_new_short = size if isinstance(size, int) else size[0]

            if short == requested_new_short:
                return image

            new_short, new_long = requested_new_short, int(requested_new_short * long / short)

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            size = (new_short, new_long) if width <= height else (new_long, new_short)

    return image.resize(size, resample=resample)

def center_crop( image, size):
    """
    Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
    size given, it will be padded (so the returned result has the size asked).
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape (n_channels, height, width) or (height, width, n_channels)):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            The size to which crop the image.
    Returns:
        new_image: A center cropped `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape: (n_channels,
        height, width).
    """
    _ensure_format_supported(image)

    if not isinstance(size, tuple):
        size = (size, size)

    # PIL Image.size is (width, height) but NumPy array and torch Tensors have (height, width)
    if is_torch_tensor(image) or isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = expand_dims(image)
        image_shape = image.shape[1:] if image.shape[0] in [1, 3] else image.shape[:2]
    else:
        image_shape = (image.size[1], image.size[0])

    top = (image_shape[0] - size[0]) // 2
    bottom = top + size[0]  # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    left = (image_shape[1] - size[1]) // 2
    right = left + size[1]  # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.

    # For PIL Images we have a method to crop directly.
    if isinstance(image, PIL.Image.Image):
        return image.crop((left, top, right, bottom))

    # Check if image is in (n_channels, height, width) or (height, width, n_channels) format
    channel_first = True if image.shape[0] in [1, 3] else False

    # Transpose (height, width, n_channels) format images
    if not channel_first:
        if isinstance(image, np.ndarray):
            image = image.transpose(2, 0, 1)
        if is_torch_tensor(image):
            image = image.permute(2, 0, 1)

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= image_shape[0] and left >= 0 and right <= image_shape[1]:
        return image[..., top:bottom, left:right]

    # Otherwise, we may need to pad if the image is too small. Oh joy...
    new_shape = image.shape[:-2] + (max(size[0], image_shape[0]), max(size[1], image_shape[1]))
    if isinstance(image, np.ndarray):
        new_image = np.zeros_like(image, shape=new_shape)
    elif is_torch_tensor(image):
        new_image = image.new_zeros(new_shape)

    top_pad = (new_shape[-2] - image_shape[0]) // 2
    bottom_pad = top_pad + image_shape[0]
    left_pad = (new_shape[-1] - image_shape[1]) // 2
    right_pad = left_pad + image_shape[1]
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    new_image = new_image[
        ..., max(0, top) : min(new_image.shape[-2], bottom), max(0, left) : min(new_image.shape[-1], right)
    ]

    return new_image

def flip_channel_order(image):
    """
    Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
    `image` to a NumPy array if it's a PIL Image.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should
            be first.
    """
    _ensure_format_supported(image)

    if isinstance(image, PIL.Image.Image):
        image = to_numpy_array(image)

    return image[::-1, :, :]

def rotate(image, angle, resample=PIL.Image.NEAREST, expand=0, center=None, translate=None, fillcolor=None):
    """
    Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
    counter clockwise around its centre.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
            rotating.
    Returns:
        image: A rotated `PIL.Image.Image`.
    """
    _ensure_format_supported(image)

    if not isinstance(image, PIL.Image.Image):
        image = to_pil_image(image)

    return image.rotate(
        angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
    )

XFund_label2ids = {
    "O":0,
    'B-HEADER':1,
    'I-HEADER':2,
    'B-QUESTION':3,
    'I-QUESTION':4,
    'B-ANSWER':5,
    'I-ANSWER':6,
}

class xfund_dataset(Dataset):
    def box_norm(self, box, width, height):
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        x0, y0, x1, y1 = box
        x0 = clip(0, int((x0 / width) * 1000), 1000)
        y0 = clip(0, int((y0 / height) * 1000), 1000)
        x1 = clip(0, int((x1 / width) * 1000), 1000)
        y1 = clip(0, int((y1 / height) * 1000), 1000)
        assert x1 >= x0
        assert y1 >= y0
        return [x0, y0, x1, y1]

    def get_segment_ids(self, bboxs):
        segment_ids = []
        for i in range(len(bboxs)):
            if i == 0:
                segment_ids.append(0)
            else:
                if bboxs[i - 1] == bboxs[i]:
                    segment_ids.append(segment_ids[-1])
                else:
                    segment_ids.append(segment_ids[-1] + 1)
        return segment_ids

    def get_position_ids(self, segment_ids):
        position_ids = []
        for i in range(len(segment_ids)):
            if i == 0:
                position_ids.append(2)
            else:
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    position_ids.append(2)
        return position_ids

    def load_data(
            self,
            data_file,
    ):
        # re-org data format
        total_data = {"id": [], "lines": [], "bboxes": [], "ner_tags": [], "image_path": []}
        for i in range(len(data_file['documents'])):
            width, height = data_file['documents'][i]['img']['width'], data_file['documents'][i]['img'][
                'height']

            cur_doc_lines, cur_doc_bboxes, cur_doc_ner_tags, cur_doc_image_path = [], [], [], []
            for j in range(len(data_file['documents'][i]['document'])):
                cur_item = data_file['documents'][i]['document'][j]
                cur_doc_lines.append(cur_item['text'])
                cur_doc_bboxes.append(self.box_norm(cur_item['box'], width=width, height=height))
                cur_doc_ner_tags.append(cur_item['label'])
            total_data['id'] += [len(total_data['id'])]
            total_data['lines'] += [cur_doc_lines]
            total_data['bboxes'] += [cur_doc_bboxes]
            total_data['ner_tags'] += [cur_doc_ner_tags]
            total_data['image_path'] += [data_file['documents'][i]['img']['fname']]

        # tokenize text and get bbox/label
        total_input_ids, total_bboxs, total_label_ids = [], [], []
        for i in range(len(total_data['lines'])):
            cur_doc_input_ids, cur_doc_bboxs, cur_doc_labels = [], [], []
            for j in range(len(total_data['lines'][i])):
                cur_input_ids = self.tokenizer(total_data['lines'][i][j], truncation=False, add_special_tokens=False, return_attention_mask=False)['input_ids']
                if len(cur_input_ids) == 0: continue

                cur_label = total_data['ner_tags'][i][j].upper()
                if cur_label == 'OTHER':
                    cur_labels = ["O"] * len(cur_input_ids)
                    for k in range(len(cur_labels)):
                        cur_labels[k] = self.label2ids[cur_labels[k]]
                else:
                    cur_labels = [cur_label] * len(cur_input_ids)
                    cur_labels[0] = self.label2ids['B-' + cur_labels[0]]
                    for k in range(1, len(cur_labels)):
                        cur_labels[k] = self.label2ids['I-' + cur_labels[k]]
                assert len(cur_input_ids) == len([total_data['bboxes'][i][j]] * len(cur_input_ids)) == len(cur_labels)
                cur_doc_input_ids += cur_input_ids
                cur_doc_bboxs += [total_data['bboxes'][i][j]] * len(cur_input_ids)
                cur_doc_labels += cur_labels
            assert len(cur_doc_input_ids) == len(cur_doc_bboxs) == len(cur_doc_labels)
            assert len(cur_doc_input_ids) > 0

            total_input_ids.append(cur_doc_input_ids)
            total_bboxs.append(cur_doc_bboxs)
            total_label_ids.append(cur_doc_labels)
        assert len(total_input_ids) == len(total_bboxs) == len(total_label_ids)

        # split text to several slices because of over-length
        input_ids, bboxs, labels = [], [], []
        segment_ids, position_ids = [], []
        image_path = []
        for i in range(len(total_input_ids)):
            start = 0
            cur_iter = 0
            while start < len(total_input_ids[i]):
                end = min(start + 510, len(total_input_ids[i]))

                input_ids.append([self.tokenizer.cls_token_id] + total_input_ids[i][start: end] + [self.tokenizer.sep_token_id])
                bboxs.append([[0, 0, 0, 0]] + total_bboxs[i][start: end] + [[1000, 1000, 1000, 1000]])
                labels.append([-100] + total_label_ids[i][start: end] + [-100])

                cur_segment_ids = self.get_segment_ids(bboxs[-1])
                cur_position_ids = self.get_position_ids(cur_segment_ids)
                segment_ids.append(cur_segment_ids)
                position_ids.append(cur_position_ids)
                image_path.append(os.path.join(self.args.data_dir, "images", total_data['image_path'][i]))

                start = end
                cur_iter += 1

        assert len(input_ids) == len(bboxs) == len(labels) == len(segment_ids) == len(position_ids)
        assert len(segment_ids) == len(image_path)

        res = {
            'input_ids': input_ids,
            'bbox': bboxs,
            'labels': labels,
            'segment_ids': segment_ids,
            'position_ids': position_ids,
            'image_path': image_path,
        }
        return res

    def __init__(
            self,
            args,
            tokenizer,
            mode
    ):
        self.args = args
        self.mode = mode
        self.cur_la = args.language
        self.tokenizer = tokenizer
        self.label2ids = XFund_label2ids


        # self.common_transform = Compose([
            # RandomResizedCropAndInterpolationWithTwoPic(
                # size=args.input_size, interpolation=args.train_interpolation,
            # ),
        # ])

        # self.patch_transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize(
                # mean=torch.tensor((0.5, 0.5, 0.5)),
                # std=torch.tensor((0.5, 0.5, 0.5)))
        # ])

        data_file = json.load(
            open(os.path.join(args.data_dir, "{}.{}.json".format(self.cur_la, 'train' if mode == 'train' else 'val')),
                 'r'))

        self.feature = self.load_data(data_file)

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        # attention_mask = self.feature["attention_mask"][index]
        attention_mask = [1] * len(input_ids)
        labels = self.feature["labels"][index]
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        img = pil_loader(self.feature['image_path'][index])
        # for_patches, _ = self.common_transform(img, augmentation=False)
        # patch = self.patch_transform(for_patches)
        for_patches = resize(img, size=self.args.input_size)
        patch = normalize(for_patches, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        assert len(input_ids) == len(attention_mask) == len(labels) == len(bbox) == len(segment_ids)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "bbox": bbox,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "images": patch,
        }
        return res

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
