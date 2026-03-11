import cv2
import argparse
import pickle
import datetime
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
from matplotlib import pyplot as plt


def indent(elem, level=0):
    """
    Adds indentation and newlines to XML elements for pretty-print.
    """
    i = "\t" * level  # use tab character
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = "\n" + "\t" * (level + 1)
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = "\n" + "\t" * level
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = "\n" + "\t" * level


def save_calibration_to_xml(fp, K, dist, rvecs, tvecs):
    dist = dist.flatten()
    t = tvecs[0]
    R = cv2.Rodrigues(rvecs[0])[0]
    Cw = (-R.T @ t).flatten()

    root = ET.Element("CameraCalibrationParams")

    # Intrinsics
    ET.SubElement(root, "Fx").text = str(K[0, 0])
    ET.SubElement(root, "Fy").text = str(K[1, 1])
    ET.SubElement(root, "Cx").text = str(K[0, 2])
    ET.SubElement(root, "Cy").text = str(K[1, 2])

    # Distortion
    ET.SubElement(root, "K1").text = str(dist[0])
    ET.SubElement(root, "K2").text = str(dist[1])
    ET.SubElement(root, "P1").text = str(dist[2])
    ET.SubElement(root, "P2").text = str(dist[3])
    ET.SubElement(root, "K3").text = str(dist[4])

    # Rotation (row-major)
    ET.SubElement(root, "R00").text = str(R[0, 0])
    ET.SubElement(root, "R01").text = str(R[0, 1])
    ET.SubElement(root, "R02").text = str(R[0, 2])
    ET.SubElement(root, "R10").text = str(R[1, 0])
    ET.SubElement(root, "R11").text = str(R[1, 1])
    ET.SubElement(root, "R12").text = str(R[1, 2])
    ET.SubElement(root, "R20").text = str(R[2, 0])
    ET.SubElement(root, "R21").text = str(R[2, 1])
    ET.SubElement(root, "R22").text = str(R[2, 2])

    # Camera center
    ET.SubElement(root, "CxW").text = str(Cw[0])
    ET.SubElement(root, "CyW").text = str(Cw[1])
    ET.SubElement(root, "CzW").text = str(Cw[2])

    # Pretty indent
    indent(root)

    # Write XML to file
    tree = ET.ElementTree(root)
    tree.write(fp, encoding="utf-8", xml_declaration=True)

def save_calibration_to_pkl(fp, ret, K, dist, rvecs, tvecs):
    calib_data = {
        'ret': ret,
        'K': K,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

    with open(fp, 'wb') as f:
        pickle.dump(calib_data, f)

def main(args):
    ## Calibration
    out_folder = Path(args.out_calib_folder)
    pattern_size = (args.cols, args.rows)

    extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    images_fp = list(f for ext in extensions for f in Path(args.images_folder).glob(f"*.{ext}"))

    # img_calib = cv2.imread(str(images_fp[0]), cv2.IMREAD_GRAYSCALE)
    # h, w = img_calib.shape[:2]

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    # Arrays to store points
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    for img_fp in images_fp:
        img_calib = cv2.imread(str(img_fp), cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img_calib = clahe.apply(img_calib)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        params.filterByCircularity = True
        params.minCircularity = 0.5
        detector = cv2.SimpleBlobDetector_create(params)

        background = cv2.GaussianBlur(img_calib, (171, 171), 0)
        img_bs = cv2.subtract(background, img_calib)

        img_bs = cv2.normalize(img_bs, None, 0, 255, cv2.NORM_MINMAX)
        img_bs = cv2.bitwise_not(img_bs.astype(np.uint8))

        keypoints = detector.detect(img_bs)
        img_bs_rgb = cv2.cvtColor(img_bs, cv2.COLOR_GRAY2RGB)
        for kpt in keypoints:
            cv2.circle(img_bs_rgb, (int(kpt.pt[0]), int(kpt.pt[1])), 5, (255, 0, 0), 2)

        _, bw = cv2.threshold(img_bs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, corners = cv2.findCirclesGrid(
                img_bs,
                (19, 14),
                flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                blobDetector=detector
            )
        objpoints.append(objp)
        imgpoints.append(corners)

        # Visual feedback
        cv2.drawChessboardCorners(img_bs_rgb, pattern_size, corners, ret)
        plt.imshow(img_bs_rgb)
        plt.show()


    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_bs.shape[::-1],
        None,
        None,
    )
    today_date_str = datetime.date.today().strftime('%Y-%m-%d')
    filename = f"{args.calib_prefix}_{today_date_str}"
    save_calibration_to_xml(out_folder / f"{filename}.xml", K, dist, rvecs, tvecs)
    save_calibration_to_pkl(out_folder / f"{filename}.pkl", ret, K, dist, rvecs, tvecs)

    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist)
    print("MSE :\n", ret)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("images_folder", type=str, help="Path to images folder")
    argparse.add_argument("--out_calib_folder", type=str, default="./calibrations/",
                          help="Path to calibration output folder")
    argparse.add_argument("--square_size", type=float, default=13.3,
                          help="Size of one square or distance between center (if circle grid) in mm")
    argparse.add_argument("--cols", type=int, default=19, help="Number of columns of target")
    argparse.add_argument("--rows", type=int, default=14, help="Number of rows of target")
    argparse.add_argument("--calib_prefix", type=str, default="calibration", help="Prefix for calibration output files")
    args = argparse.parse_args()

    main(args)