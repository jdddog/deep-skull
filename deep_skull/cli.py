# Copyright 2021 James Diprose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import pathlib
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import ray
import SimpleITK as sitk
from monai.transforms import AddChannel, LoadImage, Rotate, SaveImage
from monai.visualize import matshow3d
from skimage.morphology import closing, footprints
from wcmatch.pathlib import Path as WcPath

from deep_skull.actor import BrainExtractorActor
from deep_skull.masks import compute_contiguous_mask, fill_mask_holes

DICOM_EXTENSION = ".dcm"


def wait_for_tasks(task_ids, timeout=10.0):
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    results = []

    while True:
        ready_ids, not_ready_ids = ray.wait(task_ids, num_returns=len(task_ids), timeout=timeout)

        # Add the results that have completed
        for ready_id in ready_ids:
            result = ray.get(ready_id)
            results.append(result)
        task_ids = not_ready_ids

        logging.info(f"Num tasks complete: {len(results)}, num tasks waiting: {len(task_ids)}.")

        # If no tasks left then break
        if len(task_ids) == 0:
            break

    return results


def list_scan_paths(base_folder: str, pattern: str, case_ids: List[str]) -> List[str]:
    if not len(case_ids):
        # Matches scans with pattern e.g. STK1_ax_CT.nii.gz
        patterns = [f"*{pattern}"]
    else:
        # Matches multiple scans
        patterns = [f"*/{case_id}/*/*{pattern}" for case_id in case_ids]

    paths = []
    for path in WcPath(base_folder).rglob(patterns=patterns):
        paths.append(str(path))
    return paths


@click.group()
def cli():
    """The deep-skull command line tool"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)


@cli.command("extract-brain")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("scan_type", type=click.STRING)
@click.option("--case-id", type=click.STRING, multiple=True)
@click.option("--local-mode", type=click.BOOL, is_flag=True, help="Run the program serially in a single process.")
@click.option(
    "--num-workers", type=click.INT, help="The number of parallel workers to use to process the scans.", default=1
)
@click.option("--gpu", type=click.INT, default=0, help="The GPU to use when extracting data.")
@click.option("--overwrite", type=click.BOOL, is_flag=True, help="Overwrite existing scans.")
@click.option("--batch-size", type=click.INT, default=5, help="The batch size.")
@click.option("--debug", type=click.BOOL, is_flag=True, help="Debug flag.")
def extract_brain_cmd(
    path,
    scan_type: str,
    case_id,
    local_mode: bool,
    num_workers: int,
    gpu: int,
    overwrite: bool,
    batch_size: int,
    debug: bool,
):
    """Extract brains"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Extract brains")

    # Setup ray
    ray.init(local_mode=local_mode)

    # Get directories of cases to process
    case_ids = list(case_id)
    scan_paths = list_scan_paths(path, scan_type, case_ids)
    logging.info(f"Scans to process: {len(scan_paths)}")
    logging.info(f"Scans to process: {scan_paths}")

    # Create workers
    logging.info(f"Num workers: {num_workers}")
    workers = []
    for i in range(num_workers):
        workers.append(BrainExtractorActor.remote(local_mode, gpu, batch_size, debug=debug))
        logging.info(f"Creating worker: {i + 1}")

    # Create tasks
    task_ids = []
    i = 0
    for scan_path in scan_paths:
        # Get file name
        file_name = os.path.basename(scan_path).replace(".nii.gz", "")

        # Make mask path
        mask_path = str(pathlib.Path(*pathlib.Path(scan_path).parts[:-1], f"{file_name}_ct_bet.nii.gz").resolve())

        # Only process if output file doesn't exist or if force is True
        if not os.path.exists(mask_path) or overwrite:
            worker = workers[i % num_workers]
            task_id = worker.extract_brain.remote(scan_path, mask_path, scan_type)
            task_ids.append(task_id)
            logging.warning(f"Creating task {scan_path}, {mask_path}")
            i += 1

    # Wait for tasks to complete
    wait_for_tasks(task_ids)
    logging.info("Complete")


def combine_masks(pattern, base_folder, file_name, ct_bet_path, fsl_bet_path, debug, every_n):
    img_ct_bet, meta_ct_bet = LoadImage()(ct_bet_path)
    img_fsl_bet, meta_fsl_bet = LoadImage()(fsl_bet_path)

    # Move Z axis from 2 to 0
    img_ct_bet = np.moveaxis(img_ct_bet, source=2, destination=0)
    img_fsl_bet = np.moveaxis(img_fsl_bet, source=2, destination=0)

    # Visualise
    if debug:
        matshow3d(title=f"{file_name} ct_bet", volume=img_ct_bet, every_n=every_n, show=True, cmap="gray")
        matshow3d(title=f"{file_name} fsl_bet", volume=img_fsl_bet, every_n=every_n, show=True, cmap="gray")

    # Combine masks
    mask = np.logical_or(img_ct_bet, img_fsl_bet)
    mask = np.array(mask, dtype=np.int8)

    if debug:
        matshow3d(title=f"{file_name} combined before fill holes", volume=mask, every_n=every_n, show=True, cmap="gray")

    # Fill holes
    if "ax_A" in pattern:
        mask = fill_mask_holes(mask=mask)
        # radius 3 fills slightly more holes but is slower
        mask = closing(mask, footprint=footprints.ball(radius=2))

    if debug:
        matshow3d(title=f"{file_name} combined after fill", volume=mask, every_n=every_n, show=True, cmap="gray")

    mask = np.moveaxis(mask, source=0, destination=2)

    # Add channel as SaveImage / NiftiSaver requires it
    mask = AddChannel()(mask)

    # Save mask
    filename_or_obj = f"{file_name}_combined_bet.nii.gz"
    meta_fsl_bet["filename_or_obj"] = filename_or_obj
    SaveImage(output_dir=base_folder, output_postfix="", separate_folder=False)(mask, meta_fsl_bet)
    logging.info(f"Saved combined mask to: {os.path.join(base_folder, filename_or_obj)}")


@cli.command("combine-masks")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("pattern", type=click.STRING)
@click.option("--debug", type=click.BOOL, is_flag=True, help="Debug flag.")
def combine_masks_cmd(path, pattern: str, debug: bool):
    every_n = 5

    # Collate scans to process
    paths = []
    for p in Path(path).rglob(f"*{pattern}"):
        p = str(p.resolve())
        base_folder = os.path.dirname(p)
        file_name = os.path.basename(p).replace(".nii.gz", "")

        ct_bet_file_name = f"{file_name}_ct_bet.nii.gz"
        fsl_bet_file_name = f"{file_name}_fsl_bet.nii.gz"

        ct_bet_path = os.path.join(base_folder, ct_bet_file_name)
        fsl_bet_path = os.path.join(base_folder, fsl_bet_file_name)

        if not os.path.isfile(ct_bet_path):
            logging.warning(f"Skipping as ct_bet_path={ct_bet_path} does not exists")
        elif not os.path.isfile(fsl_bet_path):
            logging.warning(f"Skipping as fsl_bet_path={fsl_bet_path} does not exists")
        else:
            paths.append((base_folder, file_name, ct_bet_path, fsl_bet_path))

    # Combine scans
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for base_folder, file_name, ct_bet_path, fsl_bet_path in paths:
            futures.append(
                executor.submit(
                    combine_masks, pattern, base_folder, file_name, ct_bet_path, fsl_bet_path, debug, every_n
                )
            )
        for future in as_completed(futures):
            future.result()


def crop_to_mask(base_folder, file_name, img_path, ct_bet_path, debug, every_n):
    img, meta = LoadImage()(img_path)
    mask, meta_mask = LoadImage()(ct_bet_path)

    # Move Z axis from 2 to 0
    img = np.moveaxis(img, source=2, destination=0)
    mask = np.moveaxis(mask, source=2, destination=0)

    # Visualise
    if debug:
        matshow3d(title=f"{file_name} img", volume=img, every_n=every_n, show=True, cmap="gray")
        matshow3d(title=f"{file_name} mask", volume=mask, every_n=every_n, show=True, cmap="gray")

    # Crop in Z direction
    _, bbox = compute_contiguous_mask(mask=mask)
    minr, minc, minz, maxr, maxc, maxz = bbox

    # Add 5% to top and bottom
    offset = math.ceil((maxr - minr) * 0.05)
    minr = max((minr - offset), 0)
    maxr = min((maxr + offset), img.shape[0])

    img_crop = img[minr:maxr, :, :]
    mask_crop = mask[minr:maxr, :, :]

    if debug:
        matshow3d(title=f"{file_name} img cropped", volume=img_crop, every_n=every_n, show=True, cmap="gray")
        matshow3d(title=f"{file_name} mask cropped", volume=mask_crop, every_n=every_n, show=True, cmap="gray")

    # Move axes back
    img_crop = np.moveaxis(img_crop, source=0, destination=2)
    mask_crop = np.moveaxis(mask_crop, source=0, destination=2)

    # Add channel as SaveImage / NiftiSaver requires it
    img_crop = AddChannel()(img_crop)
    mask_crop = AddChannel()(mask_crop)

    # Save cropped image
    filename_or_obj = f"{file_name}_cropped"
    meta["filename_or_obj"] = filename_or_obj
    SaveImage(output_dir=base_folder, output_postfix="", separate_folder=False)(img_crop, meta)
    logging.info(f"Saved cropped image to: {os.path.join(base_folder, f'{filename_or_obj}.nii.gz')}")

    # Save cropped mask
    filename_or_obj = f"{file_name}_cropped_ct_bet"
    meta["filename_or_obj"] = filename_or_obj
    SaveImage(output_dir=base_folder, output_postfix="", separate_folder=False)(mask_crop, meta)
    logging.info(f"Saved cropped mask to: {os.path.join(base_folder, f'{filename_or_obj}.nii.gz')}")


@cli.command("crop-to-mask")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("scan_type", type=click.STRING)
@click.option("--debug", type=click.BOOL, is_flag=True, help="Debug flag.")
def crop_to_mask_cmd(path, scan_type: str, debug: bool):
    every_n = 5

    # Collate scans to process
    paths = []
    for p in Path(path).rglob(f"*{scan_type}.nii.gz"):
        p = str(p.resolve())
        base_folder = os.path.dirname(p)
        file_name = os.path.basename(p).split(".")[0]
        ct_bet_path = os.path.join(base_folder, f"{file_name}_ct_bet.nii.gz")

        if not os.path.isfile(ct_bet_path):
            logging.warning(f"Skipping as ct_bet_path={ct_bet_path} does not exists")
        else:
            paths.append((base_folder, file_name, p, ct_bet_path))

    # Combine scans
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for base_folder, file_name, img_path, ct_bet_path in paths:
            futures.append(executor.submit(crop_to_mask, base_folder, file_name, img_path, ct_bet_path, debug, every_n))
        for future in as_completed(futures):
            future.result()


def resample(
    input_path: str, output_path: str, new_spacing: Tuple[float, float, float] = (1, 1, 1), interpolator=sitk.sitkLinear
):
    print(f"Resampling {input_path} to {new_spacing[0]}x{new_spacing[1]}x{new_spacing[2]}mm")
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    image_spacing = image.GetSpacing()
    image_size = image.GetSize()
    new_size = [
        int(round(image_size_ * image_spacing_ / new_spacing_))
        for image_size_, image_spacing_, new_spacing_ in zip(image_size, image_spacing, new_spacing)
    ]

    new_image = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )
    # useCompression
    sitk.WriteImage(new_image, output_path, useCompression=True, compressionLevel=9)
    print(f"Saved to: {output_path}")


@cli.command("resample")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("pattern", type=click.STRING)
@click.option("--x", type=click.FLOAT, default=1.0, help="The resolution to resample X, in mm.")
@click.option("--y", type=click.FLOAT, default=1.0, help="The resolution to resample Y, in mm.")
@click.option("--z", type=click.FLOAT, default=1.0, help="The resolution to resample X, in mm.")
def resample_cmd(path, pattern: str, x: float, y: float, z: float):
    # Collate scans to process
    paths = []
    for p in Path(path).rglob(f"*{pattern}"):
        input_path = str(p.resolve())
        base_folder = os.path.dirname(p)
        file_name = os.path.basename(p).replace(".nii.gz", "")
        # file_name = os.path.basename(p).split(".")[0] # this line was active instead of the above when we ran the
        # experiments, which truncated some of the template names

        output_path = os.path.join(base_folder, f"{file_name}_{x}x{y}x{z}mm.nii.gz")
        paths.append((input_path, output_path))

    # Process scans
    resample_shape = (x, y, z)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for input_path, output_path in paths:
            futures.append(executor.submit(resample, input_path, output_path, new_spacing=resample_shape))
        for future in as_completed(futures):
            future.result()


def fetch_pixdim(image_path: str):
    print(f"Loading: {image_path}")
    _, meta = LoadImage()(image_path)
    pixdim = meta["pixdim"]
    x, y, z = pixdim[1], pixdim[2], pixdim[3]
    return x, y, z


@cli.command("pixdim-stats")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument("pattern", type=click.STRING)
def pixdim_stats_cmd(path, pattern: str):
    # Get metadata
    x_list, y_list, z_list = [], [], []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for input_path in Path(path).rglob(f"*ax_A.nii.gz"):
            futures.append(executor.submit(fetch_pixdim, str(input_path.resolve())))
        for input_path in Path(path).rglob(f"*ax_CT.nii.gz"):
            futures.append(executor.submit(fetch_pixdim, str(input_path.resolve())))

        for future in as_completed(futures):
            x, y, z = future.result()
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)

    print(f"Median pixdim for {pattern}")
    print(f"  x: {np.median(x_list):.4f}")
    print(f"  y: {np.median(y_list):.4f}")
    print(f"  z: {np.median(z_list):.4f}")

    print(f"Min pixdim for {pattern}")
    print(f"  x: {np.min(x_list):.4f}")
    print(f"  y: {np.min(y_list):.4f}")
    print(f"  z: {np.min(z_list):.4f}")

    print(f"Max pixdim for {pattern}")
    print(f"  x: {np.max(x_list):.4f}")
    print(f"  y: {np.max(y_list):.4f}")
    print(f"  z: {np.max(z_list):.4f}")


@cli.command("rotate")
@click.argument("file-path", type=click.Path(file_okay=True, dir_okay=False))
@click.argument("angle", type=click.FLOAT)
@click.option("--debug", type=click.BOOL, is_flag=True, help="Debug flag.")
def rotate_cmd(file_path, angle, debug: bool):
    base_folder = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).replace(".nii.gz", "")

    image, meta = LoadImage()(file_path)
    image = np.moveaxis(image, source=2, destination=0)

    if debug:
        matshow3d(title=f"Before rotation: {file_name}", volume=image, every_n=5, show=True, cmap="gray")

    # Rotate
    angle_radians = angle * math.pi / 180
    image = Rotate(angle=angle_radians, keep_size=True)(image)

    if debug:
        matshow3d(title=f"After rotation: {file_name}", volume=image, every_n=5, show=True, cmap="gray")

    image = np.moveaxis(image, source=0, destination=2)

    # Add channel as SaveImage / NiftiSaver requires it
    image = AddChannel()(image)

    # Save mask
    meta["filename_or_obj"] = f"{file_name}_rot"
    SaveImage(output_dir=base_folder, output_postfix="", separate_folder=False)(image, meta)


@cli.command("crop")
@click.argument("file-path", type=click.Path(file_okay=True, dir_okay=False))
@click.argument("z", type=click.INT)
@click.option("--debug", type=click.BOOL, is_flag=True, help="Debug flag.")
def crop_cmd(file_path, z, debug: bool):
    base_folder = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).replace(".nii.gz", "")

    image, meta = LoadImage()(file_path)
    image = np.moveaxis(image, source=2, destination=0)

    if debug:
        matshow3d(title=f"Before rotation: {file_name}", volume=image, every_n=5, show=True, cmap="gray")

    # Crop
    image = image[z:-1, :, :]

    if debug:
        matshow3d(title=f"After rotation: {file_name}", volume=image, every_n=5, show=True, cmap="gray")

    image = np.moveaxis(image, source=0, destination=2)

    # Add channel as SaveImage / NiftiSaver requires it
    image = AddChannel()(image)

    # Save mask
    meta["filename_or_obj"] = f"{file_name}_vcrop"
    SaveImage(output_dir=base_folder, output_postfix="", separate_folder=False)(image, meta)


if __name__ == "__main__":
    cli()
