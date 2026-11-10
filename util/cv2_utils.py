import math

import cv2
import imutils
import numpy as np
from imutils import contours, perspective

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 0, 255)
LINE_COLOR = (255, 0, 0)
LABEL_COLOR = (36, 255, 12)
MIN_CONTOUR_AREA = 100
CRACK_GRAY_THRESHOLD = 40


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def put_text(img, text, org):
    cv2.putText(img, text, org, FONT, 0.45, TEXT_COLOR, 1)


def horizontal_run(mask, x, y):
    """Return (row, left, right) of the horizontal run of nonzero pixels in
    `mask` that contains the nonzero pixel closest to (x, y), or None if the
    mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    i = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
    row, col = ys[i], xs[i]
    line = mask[row]
    left = col
    while left > 0 and line[left - 1]:
        left -= 1
    right = col
    while right < len(line) - 1 and line[right + 1]:
        right += 1
    return row, left, right


def crack_bbox(fused):
    """Bounding box (y1, x1, y2, x2) of all crack pixels in the fused map,
    or (0, 0, 0, 0) if there are none."""
    mask = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY) >= CRACK_GRAY_THRESHOLD
    cols = np.where(np.any(mask, axis=0))[0]
    rows = np.where(np.any(mask, axis=1))[0]
    if not cols.shape[0]:
        return 0, 0, 0, 0
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def getContours(npImage, overlay_img, realHeight, realWidth, unit, confidence, angle_th=30):
    """Annotate every crack contour of the fused map `npImage` onto a copy of
    `overlay_img` (angle, length, width, category) and return that image.
    `overlay_img` itself is returned unchanged when no contours are found."""
    gray = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 50, 80)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = imutils.grab_contours(
        cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    if len(cnts) == 0:
        # No cracks: sort_contours() would raise ValueError on an empty list.
        return overlay_img
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetricWidth = realWidth / npImage.shape[1]
    y1, x1, y2, x2 = crack_bbox(npImage)
    cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    orig = overlay_img.copy()
    cats = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = perspective.order_points(np.array(box, dtype="int"))
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        left_mid = tuple(int(v) for v in midpoint(tl, bl))
        right_mid = tuple(int(v) for v in midpoint(tr, br))

        cv2.line(orig, left_mid, right_mid, LINE_COLOR, 1)

        top_p = min([left_mid, right_mid], key=lambda p: p[1])
        bot_p = max([left_mid, right_mid], key=lambda p: p[1])
        D_ad = ((top_p[1] - bot_p[1]) ** 2 + (top_p[0] - bot_p[0]) ** 2) ** 0.5 + 1e-7
        P1 = min(top_p, bot_p, key=lambda p: p[0])
        P2 = max(top_p, bot_p, key=lambda p: p[0])
        dx = (P2[0] - P1[0]) or 1e-7
        slope = (P1[1] - P2[1]) / dx
        if slope > 0:
            angle = np.arccos((top_p[0] - bot_p[0]) / D_ad) * 180 / math.pi
            put_text(orig, "angle={:.1f}".format(angle), (max(top_p[0] - 100, 0), top_p[1] + 15))
        else:
            angle = np.arccos((bot_p[0] - top_p[0]) / D_ad) * 180 / math.pi
            put_text(orig, "angle={:.1f}".format(angle), (top_p[0], top_p[1] + 15))

        length = cv2.arcLength(c, True) / 2. * pixelsPerMetricWidth
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Measure width only on this contour's own pixels, not the whole row.
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [c], -1, 255, thickness=cv2.FILLED)
        contour_mask[gray < CRACK_GRAY_THRESHOLD] = 0
        width = 0
        run = horizontal_run(contour_mask, cX, cY)
        if run is not None:
            row, left_most_x, right_most_x = run
            width = right_most_x - left_most_x + 1
            cv2.line(orig, (int(left_most_x), int(row)), (int(right_most_x), int(row)),
                     LINE_COLOR, 1)
        width *= pixelsPerMetricWidth

        horizontal = angle < angle_th
        text_y = int(tltrY) + (40 if horizontal else 0)
        put_text(orig, "L={:.1f}".format(length) + unit, (int(tltrX), text_y))
        put_text(orig, "W={:.1f}".format(width) + unit, (int(tltrX), text_y + 15))
        cats.append(('H' if horizontal else 'V') + ('L' if slope > 0 else 'R'))
    if cats:
        cv2.putText(orig, "cat=" + ",".join(cats) + " Crack percentage={:.2f}".format(confidence.item()),
                    (x1, max(0, y1 - 5)), FONT, 0.45, LABEL_COLOR, 1)
    return orig
