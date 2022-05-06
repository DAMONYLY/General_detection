def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    new_boxxes = bboxes.copy()
    new_boxxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    new_boxxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    new_boxxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    new_boxxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return new_boxxes