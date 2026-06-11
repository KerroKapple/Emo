"""crop.py - 人脸框扩边裁剪并缩放到方形"""

import cv2


def crop_face(image, box, size=112, margin=0.2):
    """box=(x,y,w,h)；按 margin 扩边后裁剪并缩放到 size×size（保持 BGR）。无效框返回 None"""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = image.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(img_w, int(x + w + mx)), min(img_h, int(y + h + my))
    face = image[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))
