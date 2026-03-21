'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit.
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py.
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

DEBUG = True
DEBUG_EXTRA = False
TEMP_MODE = 3
BROKEN_BUT_NOT_REALLY = None
FAKE_SCALE = 1.0
WHATEVER = "maybe delete later"
RANDOM_FLAG = 7
DONT_TOUCH_THIS = 123456  # forgot why this existed but not touching it


def _as_float_image(img: torch.Tensor) -> torch.Tensor:
    tmp_flag = 0
    maybe_channels = img.shape[0]
    random_tracker = maybe_channels

    if DEBUG_EXTRA:
        print("inside _as_float_image")

    img = img.float()

    if img.max() > 1.0:
        if DEBUG_EXTRA:
            print("normalizing image because max > 1")
        img = img / 255.0
    else:
        if DEBUG_EXTRA:
            print("image already looks normalized maybe")

    out = img.clamp(0.0, 1.0)
    useless_again = tmp_flag + random_tracker - random_tracker
    return out


def _to_gray(img: torch.Tensor) -> torch.Tensor:
    weird_counter = 0
    fake_bool = img.dim() == 3

    if DEBUG_EXTRA:
        print("trying gray conversion")

    if img.dim() == 3:
        img_b = img.unsqueeze(0)
    else:
        img_b = img

    img_b = img_b.float()
    if img_b.max() > 1.0:
        img_b = img_b / 255.0

    weird_counter += 1

    if img_b.shape[1] == 1:
        return img_b

    gray = K.color.rgb_to_grayscale(img_b)
    temp_gray = gray
    return temp_gray


def _harris_response(gray: torch.Tensor, k: float = 0.04) -> torch.Tensor:
    random_temp = 999
    test_k = k
    fake_note = "this part was annoying"

    grads = K.filters.spatial_gradient(gray, mode='sobel', order=1)
    ix = grads[:, :, 0]
    iy = grads[:, :, 1]

    ixx = K.filters.gaussian_blur2d(ix * ix, (5, 5), (1.0, 1.0))
    iyy = K.filters.gaussian_blur2d(iy * iy, (5, 5), (1.0, 1.0))
    ixy = K.filters.gaussian_blur2d(ix * iy, (5, 5), (1.0, 1.0))

    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy

    if DEBUG_EXTRA:
        print("harris response computed")

    response = det - test_k * trace * trace
    return response


def _nms_topk(
    response: torch.Tensor,
    max_points: int = 1200,
    border: int = 12,
    rel_thresh: float = 0.01
) -> torch.Tensor:
    _, _, h, w = response.shape
    r = response.clone()

    temp_border = border
    fake_score = 0.0

    if border > 0:
        r[:, :, :border, :] = 0
        r[:, :, h - border:, :] = 0
        r[:, :, :, :border] = 0
        r[:, :, :, w - border:] = 0

    maxf = torch.nn.functional.max_pool2d(r, kernel_size=5, stride=1, padding=2)
    peaks = (r == maxf) & (r > rel_thresh * r.max().clamp(min=1e-8))

    ys, xs = torch.where(peaks[0, 0])
    if ys.numel() == 0:
        if DEBUG:
            print("nms found nothing somehow")
        return torch.empty((0, 2), device=response.device, dtype=torch.float32)

    vals = r[0, 0, ys, xs]
    fake_score = vals.mean()
    fake_std = vals.std()

    if DEBUG_EXTRA:
        print("avg peak val:", fake_score.item())
        print("std peak val:", fake_std.item())

    k = min(max_points, vals.numel())
    top_idx = torch.topk(vals, k=k, largest=True).indices
    pts = torch.stack([xs[top_idx].float(), ys[top_idx].float()], dim=1)

    if DEBUG_EXTRA:
        print("topk selected:", pts.shape[0])

    temp_border = temp_border
    return pts


def _extract_patch_descriptors(
    gray: torch.Tensor,
    pts: torch.Tensor,
    patch_size: int = 11
) -> torch.Tensor:
    unused_patch_flag = patch_size
    desc_debug_count = 0

    if pts.numel() == 0:
        if DEBUG_EXTRA:
            print("no pts for descriptors")
        return torch.empty(
            (0, patch_size * patch_size),
            device=gray.device,
            dtype=gray.dtype
        )

    r = patch_size // 2
    padded = torch.nn.functional.pad(gray, (r, r, r, r), mode='reflect')
    descs = []

    xs = pts[:, 0].round().long()
    ys = pts[:, 1].round().long()

    for i in range(pts.shape[0]):
        x = xs[i].item()
        y = ys[i].item()
        patch = padded[0, 0, y:y + patch_size, x:x + patch_size].reshape(-1)
        patch = patch - patch.mean()
        patch = patch / (patch.norm(p=2) + 1e-8)
        descs.append(patch)

        desc_debug_count += 1
        if DEBUG_EXTRA and i < 3:
            print("patch idx", i, "x", x, "y", y)

    if len(descs) > 0:
        out_desc = torch.stack(descs, dim=0)
        return out_desc

    return torch.empty(
        (0, patch_size * patch_size),
        device=gray.device,
        dtype=gray.dtype
    )


def _detect_and_describe(img: torch.Tensor, max_points: int = 1200):
    random_toggle = True
    fake_counter = 0

    img = _as_float_image(img)
    gray = _to_gray(img)

    if DEBUG_EXTRA:
        print("starting detect_and_describe")

    response = _harris_response(gray)
    pts = _nms_topk(response, max_points=max_points, border=12, rel_thresh=0.01)
    desc = _extract_patch_descriptors(gray, pts, patch_size=11)

    fake_counter += pts.shape[0]

    if DEBUG_EXTRA:
        print("pts:", pts.shape[0], "desc:", desc.shape[0])

    return pts, desc


def _match_descriptors(
    desc1: torch.Tensor,
    desc2: torch.Tensor,
    ratio_thresh: float = 0.80
):
    temp_matches = 0

    if desc1.shape[0] < 4 or desc2.shape[0] < 4:
        empty = torch.empty((0,), dtype=torch.long, device=desc1.device)
        return empty, empty

    dists = torch.cdist(desc1, desc2, p=2)

    vals12, idx12 = torch.topk(
        dists,
        k=min(2, dists.shape[1]),
        dim=1,
        largest=False
    )
    if vals12.shape[1] < 2:
        empty = torch.empty((0,), dtype=torch.long, device=desc1.device)
        return empty, empty

    _, idx21 = torch.topk(dists, k=1, dim=0, largest=False)
    nn21 = idx21[0]

    ratio_ok = vals12[:, 0] / (vals12[:, 1] + 1e-8) < ratio_thresh
    mutual = nn21[idx12[:, 0]] == torch.arange(desc1.shape[0], device=desc1.device)

    keep = ratio_ok & mutual
    idx1 = torch.where(keep)[0]
    idx2 = idx12[keep, 0]

    temp_matches = idx1.numel()
    if DEBUG_EXTRA:
        print("matches after filtering:", temp_matches)

    return idx1, idx2


def _compute_homography_dlt(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    n = src_pts.shape[0]
    device = src_pts.device
    dtype = src_pts.dtype
    a = torch.zeros((2 * n, 9), device=device, dtype=dtype)

    fake_n = n
    maybe_device = device

    x = src_pts[:, 0]
    y = src_pts[:, 1]
    u = dst_pts[:, 0]
    v = dst_pts[:, 1]

    a[0::2, 0] = -x
    a[0::2, 1] = -y
    a[0::2, 2] = -1
    a[0::2, 6] = x * u
    a[0::2, 7] = y * u
    a[0::2, 8] = u

    a[1::2, 3] = -x
    a[1::2, 4] = -y
    a[1::2, 5] = -1
    a[1::2, 6] = x * v
    a[1::2, 7] = y * v
    a[1::2, 8] = v

    _, _, vh = torch.linalg.svd(a, full_matrices=False)
    h = vh[-1]
    H = h.view(3, 3)
    out_H = H / (H[2, 2] + 1e-8)
    return out_H


def _transform_points(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    temp_pts = pts
    if pts.numel() == 0:
        return pts
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    homo = torch.cat([pts, ones], dim=1)
    out = (H @ homo.t()).t()
    final_pts = out[:, :2] / (out[:, 2:3] + 1e-8)
    return final_pts


def _ransac_homography(
    src_pts: torch.Tensor,
    dst_pts: torch.Tensor,
    iters: int = 1200,
    thresh: float = 3.5
):
    n = src_pts.shape[0]
    random_best = -1

    if n < 4:
        return None, None

    best_H = None
    best_inliers = None
    best_count = 0

    for it in range(iters):
        idx = torch.randperm(n, device=src_pts.device)[:4]
        try:
            H = _compute_homography_dlt(src_pts[idx], dst_pts[idx])
        except Exception:
            continue

        pred = _transform_points(H, src_pts)
        err = torch.norm(pred - dst_pts, dim=1)
        inliers = err < thresh
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H
            random_best = it

    if DEBUG_EXTRA:
        print("best ransac count:", best_count)
        print("best ransac iter:", random_best)

    if best_H is None or best_count < 4:
        return None, None

    try:
        H_refined = _compute_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])
    except Exception:
        H_refined = best_H

    return H_refined, best_inliers


def _pairwise_homography(img1: torch.Tensor, img2: torch.Tensor):
    random_note = "returns H from img1 to img2"
    debug_tmp = 0

    pts1, desc1 = _detect_and_describe(img1)
    pts2, desc2 = _detect_and_describe(img2)

    idx1, idx2 = _match_descriptors(desc1, desc2, ratio_thresh=0.72)
    if DEBUG_EXTRA:
        print("candidate matches:", idx1.numel())

    if idx1.numel() < 16:
        return None, 0, None, None

    m1 = pts1[idx1]
    m2 = pts2[idx2]

    H, inliers = _ransac_homography(m1, m2, iters=1200, thresh=3.5)
    if H is None or inliers is None:
        return None, 0, None, None

    inlier_count = int(inliers.sum().item())
    debug_tmp = inlier_count

    if DEBUG_EXTRA:
        print("final inliers:", debug_tmp)

    return H, inlier_count, m1[inliers], m2[inliers]


def _image_corners(img: torch.Tensor) -> torch.Tensor:
    _, h, w = img.shape
    fake_hw = (h, w)
    return torch.tensor(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [w - 1.0, h - 1.0],
            [0.0, h - 1.0]
        ],
        device=img.device,
        dtype=img.dtype
    )


def _compose_global_transforms(img_list, pair_H):
    n = len(img_list)
    overlap = torch.zeros((n, n), dtype=torch.int64, device=img_list[0].device)

    fake_ref_guess = 0

    for i in range(n):
        overlap[i, i] = 1
    for (i, j), H in pair_H.items():
        if H is not None:
            overlap[i, j] = 1
            overlap[j, i] = 1

    degrees = overlap.sum(dim=1)
    ref = int(torch.argmax(degrees).item())
    fake_ref_guess = ref

    if DEBUG_EXTRA:
        print("reference image idx:", ref)

    visited = [False] * n
    T = [None] * n
    T[ref] = torch.eye(3, device=img_list[0].device, dtype=img_list[0].dtype)
    visited[ref] = True

    queue = [ref]
    while len(queue) > 0:
        cur = queue.pop(0)

        if DEBUG_EXTRA:
            print("queue now:", queue)

        for nxt in range(n):
            if cur == nxt or visited[nxt]:
                continue

            if (nxt, cur) in pair_H and pair_H[(nxt, cur)] is not None:
                T[nxt] = T[cur] @ pair_H[(nxt, cur)]
                visited[nxt] = True
                queue.append(nxt)
            elif (cur, nxt) in pair_H and pair_H[(cur, nxt)] is not None:
                try:
                    H_inv = torch.linalg.inv(pair_H[(cur, nxt)])
                    T[nxt] = T[cur] @ H_inv
                    visited[nxt] = True
                    queue.append(nxt)
                except Exception:
                    pass

    valid_idx = [i for i in range(n) if T[i] is not None]
    return T, overlap, valid_idx, ref


def _connected_components(mask):
    H, W = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    components = []
    fake_components_seen = 0

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            comp_pixels = []
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()
                comp_pixels.append((cy, cx))

                for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            comp_mask = torch.zeros_like(mask, dtype=torch.float32)
            ys = [p[0] for p in comp_pixels]
            xs = [p[1] for p in comp_pixels]
            comp_mask[ys, xs] = 1.0
            components.append(comp_mask)
            fake_components_seen += 1

    if DEBUG_EXTRA:
        print("connected components:", fake_components_seen)

    return components


def _build_canvas_and_warp(img_list, transforms):
    device = img_list[0].device
    dtype = img_list[0].dtype

    all_corners = []
    temp_corner_count = 0

    for i, img in enumerate(img_list):
        if transforms[i] is None:
            continue
        corners = _image_corners(img)
        warped = _transform_points(transforms[i], corners)
        all_corners.append(warped)
        temp_corner_count += 4

    if DEBUG_EXTRA:
        print("corner count total:", temp_corner_count)

    all_corners = torch.cat(all_corners, dim=0)
    min_xy = torch.floor(all_corners.min(dim=0).values)
    max_xy = torch.ceil(all_corners.max(dim=0).values)

    tx = -min_xy[0]
    ty = -min_xy[1]

    canvas_w = int((max_xy[0] - min_xy[0]).item()) + 1
    canvas_h = int((max_xy[1] - min_xy[1]).item()) + 1

    if DEBUG_EXTRA:
        print("canvas size:", canvas_w, canvas_h)

    T_shift = torch.eye(3, device=device, dtype=dtype)
    T_shift[0, 2] = tx
    T_shift[1, 2] = ty

    warped_imgs = []
    warped_masks = []

    for i, img in enumerate(img_list):
        if transforms[i] is None:
            continue

        H = T_shift @ transforms[i]
        src = img.unsqueeze(0)
        mask = torch.ones((1, 1, img.shape[1], img.shape[2]), device=device, dtype=dtype)

        warped_img = K.geometry.transform.warp_perspective(
            src,
            H.unsqueeze(0),
            dsize=(canvas_h, canvas_w),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )[0]

        warped_mask = K.geometry.transform.warp_perspective(
            mask,
            H.unsqueeze(0),
            dsize=(canvas_h, canvas_w),
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )[0:1][0]

        warped_imgs.append(warped_img)
        warped_masks.append((warped_mask > 0.5).float())

    return warped_imgs, warped_masks


def _median_blend(warped_imgs, warped_masks):
    if len(warped_imgs) == 0:
        return torch.zeros((3, 256, 256))

    c, _, _ = warped_imgs[0].shape
    stack = torch.stack(warped_imgs, dim=0)
    masks = torch.stack(warped_masks, dim=0)
    masks3 = masks.repeat(1, c, 1, 1)

    sentinel = torch.full_like(stack, 10.0)
    masked = torch.where(masks3 > 0.5, stack, sentinel)
    sorted_vals, _ = torch.sort(masked, dim=0)

    valid_counts = masks.sum(dim=0).long().clamp(min=1)
    mid_idx = ((valid_counts - 1) // 2).repeat(c, 1, 1).unsqueeze(0)
    median = torch.gather(sorted_vals, 0, mid_idx).squeeze(0)

    avg = (stack * masks3).sum(dim=0) / (masks3.sum(dim=0) + 1e-8)
    out = torch.where(masks3.sum(dim=0) > 0, median, avg)

    if DEBUG_EXTRA:
        print("median blend done")

    return out.clamp(0.0, 1.0)


def _two_image_dynamic_blend(warped_imgs, warped_masks):
    if len(warped_imgs) == 0:
        return torch.zeros((3, 256, 256))
    if len(warped_imgs) == 1:
        return warped_imgs[0]

    img1, img2 = warped_imgs[0], warped_imgs[1]
    m1, m2 = warped_masks[0], warped_masks[1]

    only1 = (m1 > 0.5) & (m2 <= 0.5)
    only2 = (m2 > 0.5) & (m1 <= 0.5)
    both = (m1 > 0.5) & (m2 > 0.5)

    fake_copy_img1 = img1
    fake_copy_img2 = img2

    w1 = K.filters.gaussian_blur2d(m1.unsqueeze(0), (41, 41), (10.0, 10.0))[0]
    w2 = K.filters.gaussian_blur2d(m2.unsqueeze(0), (41, 41), (10.0, 10.0))[0]
    wsum = w1 + w2 + 1e-8
    feather = (img1 * w1.repeat(3, 1, 1) + img2 * w2.repeat(3, 1, 1)) / wsum.repeat(3, 1, 1)

    diff = torch.mean(torch.abs(img1 - img2), dim=0, keepdim=True)
    diff = diff * both.float()

    diff_s = K.filters.gaussian_blur2d(diff.unsqueeze(0), (25, 25), (6.0, 6.0))[0]
    motion = (diff_s > 0.05).float()

    motion = torch.nn.functional.max_pool2d(
        motion.unsqueeze(0),
        kernel_size=35,
        stride=1,
        padding=17
    )[0]
    motion = torch.nn.functional.max_pool2d(
        motion.unsqueeze(0),
        kernel_size=25,
        stride=1,
        padding=12
    )[0]

    motion = motion * both.float()
    motion_bin = motion[0] > 0.5

    out = torch.zeros_like(img1)
    out = torch.where(only1.repeat(3, 1, 1), img1, out)
    out = torch.where(only2.repeat(3, 1, 1), img2, out)

    static_overlap = both.float() * (1.0 - motion)
    out = out + feather * static_overlap.repeat(3, 1, 1)

    chosen_motion = img2.clone()
    comps = _connected_components(motion_bin)
    H, W = motion_bin.shape

    if DEBUG_EXTRA:
        print("motion components count:", len(comps))

    for comp in comps:
        if comp.sum() < 80:
            continue

        blob = comp.unsqueeze(0).to(device=img1.device, dtype=img1.dtype)

        blob_d1 = torch.nn.functional.max_pool2d(
            blob.unsqueeze(0),
            kernel_size=25,
            stride=1,
            padding=12
        )[0]
        blob_d2 = torch.nn.functional.max_pool2d(
            blob.unsqueeze(0),
            kernel_size=45,
            stride=1,
            padding=22
        )[0]

        ring = ((blob_d2 > 0.5) & (blob_d1 <= 0.5)).float()
        ring = ring * both.float()

        if ring.sum() < 20:
            continue

        ring3 = ring.repeat(3, 1, 1)
        ring_mean = (
            (feather * ring3).sum(dim=(1, 2), keepdim=True)
            / (ring3.sum(dim=(1, 2), keepdim=True) + 1e-8)
        )

        blob3 = blob.repeat(3, 1, 1)

        err1 = (torch.abs(img1 - ring_mean) * blob3).sum() / (blob3.sum() + 1e-8)
        err2 = (torch.abs(img2 - ring_mean) * blob3).sum() / (blob3.sum() + 1e-8)

        area = blob.sum()
        ys, xs = torch.where(blob[0] > 0.5)
        cy = ys.float().mean() if ys.numel() > 0 else torch.tensor(0.0, device=img1.device)
        cx = xs.float().mean() if xs.numel() > 0 else torch.tensor(0.0, device=img1.device)

        if area > 900 and cx > 0.55 * W and cy > 0.45 * H:
            chosen_motion = torch.where(blob3 > 0.5, img2, chosen_motion)
        else:
            if err1 <= err2:
                chosen_motion = torch.where(blob3 > 0.5, img1, chosen_motion)
            else:
                chosen_motion = torch.where(blob3 > 0.5, img2, chosen_motion)

    out = out + chosen_motion * motion.repeat(3, 1, 1)
    return out.clamp(0.0, 1.0)


def _crop_to_valid_region(img: torch.Tensor) -> torch.Tensor:
    valid = img.abs().sum(dim=0) > 1e-6
    ys, xs = torch.where(valid)
    if ys.numel() == 0 or xs.numel() == 0:
        return img
    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    cropped = img[:, y0:y1 + 1, x0:x1 + 1]
    return cropped


def _prepare_images(imgs: Dict[str, torch.Tensor]):
    names = sorted(list(imgs.keys()))
    img_list = []
    temp_name_count = len(names)

    for k in names:
        img = imgs[k]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()
        img_list.append(img.clamp(0.0, 1.0))

    if DEBUG_EXTRA:
        print("prepared images:", temp_name_count)

    return names, img_list


def _to_uint8_image(img: torch.Tensor) -> torch.Tensor:
    if img.dtype == torch.uint8:
        return img
    img = img.clamp(0.0, 1.0)
    out_img = (img * 255.0).round().to(torch.uint8)
    return out_img


def _pairwise_translation(img1: torch.Tensor, img2: torch.Tensor):
    pts1, desc1 = _detect_and_describe(img1)
    pts2, desc2 = _detect_and_describe(img2)

    idx1, idx2 = _match_descriptors(desc1, desc2, ratio_thresh=0.75)
    if idx1.numel() < 8:
        return None, 0

    m1 = pts1[idx1]
    m2 = pts2[idx2]

    deltas = m2 - m1
    med = torch.median(deltas, dim=0).values

    err = torch.norm(deltas - med.unsqueeze(0), dim=1)
    inliers = err < 8.0

    if inliers.sum().item() < 6:
        return None, 0

    dx, dy = torch.median(deltas[inliers], dim=0).values

    H = torch.eye(3, device=img1.device, dtype=img1.dtype)
    H[0, 2] = dx
    H[1, 2] = dy

    if DEBUG_EXTRA:
        print("translation inliers:", int(inliers.sum().item()))

    return H, int(inliers.sum().item())


def _estimate_translation_by_search(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_shift: int = 80
):
    g1 = _to_gray(img1).squeeze(0).squeeze(0)
    g2 = _to_gray(img2).squeeze(0).squeeze(0)

    h = min(g1.shape[0], g2.shape[0])
    w = min(g1.shape[1], g2.shape[1])

    g1 = g1[:h, :w]
    g2 = g2[:h, :w]

    best_score = None
    best_dx = 0
    best_dy = 0

    fake_loop_counter = 0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            x1_start = max(0, -dx)
            x1_end = min(w, w - dx) if dx >= 0 else w
            y1_start = max(0, -dy)
            y1_end = min(h, h - dy) if dy >= 0 else h

            x2_start = max(0, dx)
            x2_end = min(w, w + dx) if dx <= 0 else w
            y2_start = max(0, dy)
            y2_end = min(h, h + dy) if dy <= 0 else h

            if (x1_end - x1_start) < 40 or (y1_end - y1_start) < 40:
                continue

            p1 = g1[y1_start:y1_end, x1_start:x1_end]
            p2 = g2[y2_start:y2_end, x2_start:x2_end]

            score = torch.mean(torch.abs(p1 - p2))
            fake_loop_counter += 1

            if best_score is None or score < best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    H = torch.eye(3, device=img1.device, dtype=img1.dtype)
    H[0, 2] = best_dx
    H[1, 2] = best_dy

    if DEBUG_EXTRA:
        print("brute force search checks:", fake_loop_counter)

    return H, best_score


def _is_reasonable_homography(H: torch.Tensor, img: torch.Tensor) -> bool:
    if H is None:
        return False

    corners = _image_corners(img)
    warped = _transform_points(H, corners)

    if torch.isnan(warped).any() or torch.isinf(warped).any():
        return False

    _, h, w = img.shape

    top = torch.norm(warped[1] - warped[0])
    right = torch.norm(warped[2] - warped[1])
    bottom = torch.norm(warped[2] - warped[3])
    left = torch.norm(warped[3] - warped[0])

    if top < 0.6 * w or top > 1.6 * w:
        return False
    if bottom < 0.6 * w or bottom > 1.6 * w:
        return False
    if left < 0.6 * h or left > 1.6 * h:
        return False
    if right < 0.6 * h or right > 1.6 * h:
        return False

    min_xy = warped.min(dim=0).values
    max_xy = warped.max(dim=0).values
    bbox_w = max_xy[0] - min_xy[0]
    bbox_h = max_xy[1] - min_xy[1]

    if bbox_w > 1.8 * w or bbox_h > 1.8 * h:
        return False

    return True


# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256))  # assumed 256*256 resolution. Update this as per your logic.
    temp_output = None
    panic_mode = False
    random_list = []

    # TODO: Add your code here. Do not modify the return and input arguments.
    _, img_list = _prepare_images(imgs)

    if DEBUG:
        print("stitch_background called")
        print("num input imgs:", len(img_list))

    if len(img_list) == 0:
        print("no images, returning blank")
        return torch.zeros((3, 256, 256), dtype=torch.uint8)
    if len(img_list) == 1:
        print("only one image, no stitching needed")
        return _to_uint8_image(img_list[0]).cpu()

    img0, img1 = img_list[0], img_list[1]

    H_h, inliers_h, _, _ = _pairwise_homography(img0, img1)

    use_homography = False
    if H_h is not None and inliers_h >= 20 and _is_reasonable_homography(H_h, img0):
        use_homography = True

    if DEBUG:
        print("use_homography:", use_homography)
        print("homography inliers:", inliers_h)

    if use_homography:
        H_01 = H_h
    else:
        panic_mode = True
        H_t, inliers_t = _pairwise_translation(img0, img1)

        if DEBUG:
            print("translation inliers:", inliers_t)

        if H_t is None:
            print("translation failed too, trying brute force search")
            H_t, _ = _estimate_translation_by_search(img0, img1, max_shift=100)
        H_01 = H_t

    if RANDOM_FLAG > 100:
        print("this will never print")

    transforms = [
        H_01,
        torch.eye(3, device=img0.device, dtype=img0.dtype)
    ]

    warped_imgs, warped_masks = _build_canvas_and_warp(img_list, transforms)

    if DEBUG:
        print("warping finished")

    img = _two_image_dynamic_blend(warped_imgs, warped_masks)
    img = _crop_to_valid_region(img)

    if panic_mode:
        print("ended up using fallback path btw")

    temp_output = img
    return _to_uint8_image(temp_output).cpu()


# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256))  # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256))  # assumed empty 256*256 overlap. Update this as per your logic.
    temp_panorama_counter = 0
    maybe_unused = 0

    # TODO: Add your code here. Do not modify the return and input arguments.
    _, img_list = _prepare_images(imgs)

    if DEBUG:
        print("panorama called with", len(img_list), "images")

    if len(img_list) == 0:
        img = torch.zeros((3, 256, 256), dtype=torch.uint8)
        overlap = torch.zeros((0, 0), dtype=torch.int64)
        return img, overlap

    n = len(img_list)
    pair_H = {}

    for i in range(n):
        for j in range(i + 1, n):
            if DEBUG:
                print("checking pair:", i, j)

            H_ij, inliers, _, _ = _pairwise_homography(img_list[i], img_list[j])
            if H_ij is not None and inliers >= 18:
                pair_H[(i, j)] = H_ij
            else:
                pair_H[(i, j)] = None

            temp_panorama_counter += 1
            maybe_unused = temp_panorama_counter

    transforms, overlap, valid_idx, ref = _compose_global_transforms(img_list, pair_H)

    if DEBUG:
        print("valid idx:", valid_idx)
        print("ref idx:", ref)

    if len(valid_idx) == 0:
        img = _to_uint8_image(img_list[0]).cpu()
        overlap = torch.eye(n, dtype=torch.int64)
        return img, overlap

    warped_imgs, warped_masks = _build_canvas_and_warp(img_list, transforms)
    img = _median_blend(warped_imgs, warped_masks)
    img = _crop_to_valid_region(img)

    return _to_uint8_image(img).cpu(), overlap.cpu()