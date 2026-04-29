import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import comfy.model_management
import folder_paths
import node_helpers


class LoadImageDownscale:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "load_image"

    def load_image(self, image, max_size):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        dtype = comfy.model_management.intermediate_dtype()

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            frame = i.convert("RGB")

            if len(output_images) == 0:
                w = frame.size[0]
                h = frame.size[1]

            if frame.size[0] != w or frame.size[1] != h:
                continue

            # Scale so the longest side equals max_size (up or down)
            fw, fh = frame.size
            scale = min(max_size / fw, max_size / fh)
            new_w = max(1, round(fw * scale))
            new_h = max(1, round(fh * scale))
            resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
            frame = frame.resize((new_w, new_h), resample)
            if 'A' in i.getbands():
                alpha = i.getchannel('A').resize((new_w, new_h), resample)
            elif i.mode == 'P' and 'transparency' in i.info:
                alpha = i.convert('RGBA').getchannel('A').resize((new_w, new_h), resample)
            else:
                alpha = None

            image_array = np.array(frame).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]

            if alpha is not None:
                mask = np.array(alpha).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((frame.size[1], frame.size[0]), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor.to(dtype=dtype))
            output_masks.append(mask.unsqueeze(0).to(dtype=dtype))

            if img.format == "MPO":
                break

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        out_w = output_image.shape[2]
        out_h = output_image.shape[1]

        return (output_image, output_mask, out_w, out_h)

    @classmethod
    def IS_CHANGED(s, image, max_size):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        m.update(str(max_size).encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, max_size):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODE_CLASS_MAPPINGS = {
    "LoadImageDownscale": LoadImageDownscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageDownscale": "Load Image (max Size)",
}
