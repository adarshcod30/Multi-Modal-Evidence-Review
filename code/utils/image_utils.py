"""
Image utilities — loads images as base64 for VLM API calls.
"""

import base64
import os


def load_image_as_base64(image_path: str) -> str:
    """Load an image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def parse_image_paths(image_paths_str: str) -> list:
    """Parse semicolon-separated image paths from CSV field."""
    if not image_paths_str:
        return []
    return [p.strip() for p in image_paths_str.split(";") if p.strip()]


def get_image_id(image_path: str) -> str:
    """Extract image ID (filename without extension) from path."""
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    return name


def get_media_type(image_path: str) -> str:
    """Determine media type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")
