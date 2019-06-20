import numpy as np

def nms(rects, contours, overlapThresh):
    boxes = []
    result_1 = []
    for x, y, w, h in rects:
        boxes.append([x, y, x + w, y + h])
    boxes = np.array(boxes)
    contours = np.array(contours)
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    boxes = boxes[pick].astype("int").tolist()
    result_2 = contours[pick].tolist()
    for x, y, x1, y1 in boxes:
        result_1.append([x, y, x1 - x, y1 - y])
    return [result_1, result_2]