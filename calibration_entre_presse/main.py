import argparse

import cv2
import numpy as np
import glob
from pathlib import Path

import mil as MIL

import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from tqdm import tqdm
from datetime import datetime

import pickle

matplotlib.use('TkAgg')
plt.ion()

def pixels_to_world(uv_array, K, dist, rvecs, tvecs, view_idx):
    """
    uv_array: Nx2 array of (u, v) pixel coordinates
    view_idx: index into rvecs/tvecs for the current camera pose
    Returns: Nx3 array of (X, Y, Z) in world coordinates (Z will be 0)
    """
    rvec = rvecs[view_idx]
    tvec = tvecs[view_idx]

    # 1. Undistort all pixels to normalized camera coordinates
    uv_array = np.asarray(uv_array, dtype=np.float32).reshape(-1, 1, 2)
    uv_undist = cv2.undistortPoints(uv_array, K, dist)  # shape (N,1,2)

    # 2. Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)

    # 3. Invert extrinsics to get camera position in world coords
    R_inv = R.T
    cam_pos = -R_inv @ tvec.reshape(3, 1)  # (3,1)

    # 4. Ray directions in world coords
    N = uv_undist.shape[0]
    rays_cam = np.hstack([uv_undist.reshape(N, 2), np.ones((N, 1), np.float32)])  # (N,3)
    rays_world = (R_inv @ rays_cam.T).T  # (N,3)

    # 5. Intersect each ray with Z=0 plane
    s = -cam_pos[2, 0] / rays_world[:, 2]
    Pw = cam_pos.T + (s[:, None] * rays_world)  # (N,3)

    return Pw  # Nx3 array

def keep_first_in_radius(points, r):
    """
    Keep only the first occurrence of points that are within radius r of each other.

    Parameters
    ----------
    points : (N, D) array
        Input points (image or world coords).
    r : float
        Radius threshold.

    Returns
    -------
    kept_points : (M, D) array
        Filtered points.
    kept_idx : (M,) array
        Indices of kept points in the original array.
    """
    pts = np.asarray(points, dtype=float)
    N = len(pts)

    tree = cKDTree(pts)
    visited = np.zeros(N, dtype=bool)
    kept_idx = []

    for i in range(N):
        if visited[i]:
            continue
        kept_idx.append(i)
        # mark all neighbors within r as visited
        neighbors = tree.query_ball_point(pts[i], r)
        visited[neighbors] = True

    kept_idx = np.array(kept_idx, dtype=int)
    return kept_idx

def main(args):
    date_str = datetime.now().strftime("%Y_%m_%d")

    img_folder = Path(args.images_folder)
    out_folder = Path(args.out_calib_folder)
    pattern_size = (args.cols, args.rows)

    if not out_folder.is_dir():
        out_folder.mkdir(parents=True, exist_ok=True)

    # Prepare object points for the grid (same for all images)
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.stack(np.meshgrid(np.arange(args.cols), np.arange(args.rows)), axis=-1).reshape(-1, 2).astype(
        np.float32)
    objp *= args.square_size

    flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 120
    params.thresholdStep = 8

    params.filterByArea = True
    params.minArea = 50  # adjust: ~0.5× circle area
    params.maxArea = 15000  # adjust: ~2× circle area

    params.filterByCircularity = True
    params.minCircularity = 0.75  # circles only

    params.filterByInertia = True
    params.minInertiaRatio = 0.65  # more roundness constraint

    params.filterByConvexity = True
    params.minConvexity = 0.85

    params.filterByColor = True
    params.blobColor = 0

    detector = cv2.SimpleBlobDetector_create(params)

    # Arrays to store points
    objpoints = []  # 3D points (in target coords)
    imgpoints = []  # 2D points (detected in image)

    # Load all calibration images
    extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    images = [f for ext in extensions for f in img_folder.rglob(f"*.{ext}")]

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    for fname in tqdm(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findCirclesGrid(gray, pattern_size, blobDetector=detector, flags=flags)
        if not ret:
            _, mask = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
            ret, corners = cv2.findCirclesGrid(mask, pattern_size, blobDetector=detector, flags=flags)

        if not ret:
            #  Equalize histogram
            for i in range(1, 6, 1):
                clahe = cv2.createCLAHE(clipLimit=float(i), tileGridSize=(31, 31))
                gray = clahe.apply(gray)

                # gray = cv2.medianBlur(gray, 3)
                ret, corners = cv2.findCirclesGrid(gray, pattern_size, blobDetector=detector, flags=flags)
                if ret:
                    break

        if ret:
            img_out = cv2.drawChessboardCorners(img, pattern_size, corners, True)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Failed to find corners in {fname}")
            plt.imshow(img)
            plt.show()

    # Calibrate camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints, gray.shape[::-1],
        None, None,
        flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_FIX_ASPECT_RATIO
    )

    fp_cv2_pkl_calib = out_folder / f"{args.calib_prefix}_{date_str}.pkl"
    with fp_cv2_pkl_calib.open("wb") as f:
        pickle.dump({
            "ret": ret,
            "K": K,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs
        }, f)
    print(f"OPENCV - MSE calibration: {ret}")

    pts_world_all = []
    for i in range(len(imgpoints)):
        if len(imgpoints[i]) != 0:
            pts_world_all.append(pixels_to_world(imgpoints[i], K, dist, rvecs, tvecs, 0))

    pts_world_all = np.concatenate(pts_world_all, axis=0).astype(np.float64)
    imgpoints = np.concatenate(imgpoints, axis=0).reshape(-1, 2).astype(np.float64)

    keep_idx = keep_first_in_radius(pts_world_all, r=2.0)
    imgpoints, pts_world_all = imgpoints[keep_idx], pts_world_all[keep_idx]

    x_world_all, y_world_all = np.ascontiguousarray(pts_world_all[:, 0]), np.ascontiguousarray(pts_world_all[:, 1])
    x_pix_all, y_pix_all = np.ascontiguousarray(imgpoints[:, 0]), np.ascontiguousarray(imgpoints[:, 1])
    z_world_all = np.zeros_like(x_world_all)
    nb_points = len(x_pix_all)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x_world_all, y_world_all, np.zeros_like(y_world_all))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Interactive World 3D Plot')
    plt.show(block=True)

    # MIL calibration
    app, sys, display = MIL.MappAllocDefault(MIL.M_DEFAULT, DigIdPtr=MIL.M_NULL, ImageBufIdPtr=MIL.M_NULL)
    cal_ctx = MIL.McalAlloc(sys, MIL.M_LINEAR_INTERPOLATION, MIL.M_DEFAULT, MIL.M_NULL)
    MIL.McalList(
        cal_ctx,
        x_pix_all,
        y_pix_all,
        x_world_all,
        y_world_all,
        z_world_all,
        nb_points,
        MIL.M_DEFAULT,
        MIL.M_DEFAULT
    )
    pix_error = MIL.McalInquire(cal_ctx, MIL.M_AVERAGE_PIXEL_ERROR + MIL.M_TYPE_MIL_FLOAT)
    world_error = MIL.McalInquire(cal_ctx, MIL.M_AVERAGE_WORLD_ERROR + MIL.M_TYPE_MIL_FLOAT)
    print(f"MIL - Pixel error : {pix_error:.4f} - World error : {world_error:.4f}")

    fp_mil_mca_calib = out_folder / f"{args.calib_prefix}_{date_str}.pkl"
    MIL.McalSave(str(fp_mil_mca_calib.resolve()), cal_ctx, MIL.M_DEFAULT)
    dst_img = MIL.MbufAlloc2d(sys, w, h, 8, MIL.M_IMAGE + MIL.M_PROC + MIL.M_DISP)
    dst_img_np = np.zeros((h, w), dtype=np.uint8)
    MIL.MbufPut2d(dst_img, 0, 0, w, h, dst_img_np)

    src_img = MIL.MbufRestore(str(images[0].resolve()), sys, MIL.M_NULL)
    MIL.McalTransformImage(src_img, dst_img, cal_ctx, MIL.M_DEFAULT, MIL.M_DEFAULT, MIL.M_DEFAULT)
    MIL.MdispSelect(display, dst_img)

    input("Press Enter to continue...")
    MIL.MbufFree(src_img)
    MIL.MbufFree(dst_img)
    MIL.McalFree(cal_ctx)
    MIL.MdispFree(display)
    MIL.MsysFree(sys)
    MIL.MappFree(app)

# ROW, COL = 16, 23
# pattern_size = (COL, ROW)   # checkerboard corners
# square_size = 35     # mm
# scale_factor = 1.0
# sym_grid = True
# IMG_FOLDER = "C:\\Users\\alexis.desgagne\\PycharmProjects\\Script_Ascenta\\calibration_entre_presse\\images\\*.bmp" #"calibration_images/botteuse/*.jpg"
# out_calib_name = "calibration_"

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("images_folder", type=str, help="Path to images folder")
    argparse.add_argument("--out_calib_folder", type=str, default="./calibrations/",
                          help="Path to calibration output folder")
    argparse.add_argument("--square_size", type=float, default=35,
                          help="Size of one square or distance between center (if circle grid) in mm")
    argparse.add_argument("--cols", type=int, default=23, help="Number of columns of target")
    argparse.add_argument("--rows", type=int, default=16, help="Number of rows of target")
    argparse.add_argument("--calib_prefix", type=str, default="calibration", help="Prefix for calibration output files")
    args = argparse.parse_args()

    main(args)