from sklearn.neighbors import KDTree
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path

input_file_path  = '/home/ajai-krishna/work/GEO_AI/outputs/subsets/gujarath_dtm_RGBZ.tif'
output_file_path = '/home/ajai-krishna/work/GEO_AI/outputs/gujarath_dtm_RGBZ_filled.tif'  # ← filename added

def fill_nodata_knn_chunkwise(input_tif, output_tif, chunk_size=1024, k=12, overlap=64):

    with rasterio.open(input_tif) as src:
        H       = src.height
        W       = src.width
        nodata  = src.nodata
        profile = src.profile.copy()

    print(f"Raster size  : {H} x {W}")
    print(f"Bands        : {profile['count']}")
    print(f"Nodata value : {nodata}")
    print(f"Chunk size   : {chunk_size}")
    print(f"Overlap      : {overlap}px")

    # Update output profile
    profile.update(dtype="float32", nodata=nodata)

    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    # Create empty output raster
    with rasterio.open(output_tif, "w", **profile) as dst:
        pass

    # Generate chunk indices
    y_starts = list(range(0, H, chunk_size))
    x_starts = list(range(0, W, chunk_size))

    total_chunks = len(y_starts) * len(x_starts)
    print(f"Total chunks : {total_chunks}\n")

    chunk_id = 0

    # Open files once
    with rasterio.open(input_tif) as src, rasterio.open(output_tif, "r+") as dst:

        for y in y_starts:
            for x in x_starts:

                chunk_id += 1

                # Expand window with overlap
                y_read = max(0, y - overlap)
                x_read = max(0, x - overlap)

                h_read = min(chunk_size + 2 * overlap, H - y_read)
                w_read = min(chunk_size + 2 * overlap, W - x_read)

                read_window = Window(x_read, y_read, w_read, h_read)

                chunk = src.read(window=read_window).astype(np.float32)

                # Z band
                z_band = chunk[3]

                rows, cols = np.meshgrid(
                    np.arange(z_band.shape[0]),
                    np.arange(z_band.shape[1]),
                    indexing="ij"
                )

                valid_mask  = z_band != nodata
                nodata_mask = ~valid_mask

                nodata_count = np.sum(nodata_mask)
                valid_count  = np.sum(valid_mask)

                if nodata_count == 0:

                    print(f"Chunk [{chunk_id}/{total_chunks}] y={y} x={x} → no nodata")

                elif valid_count == 0:

                    print(f"Chunk [{chunk_id}/{total_chunks}] y={y} x={x} → all nodata skipped")

                else:

                    k_use = min(k, valid_count)

                    known_pts  = np.column_stack((rows[valid_mask], cols[valid_mask]))
                    known_vals = z_band[valid_mask]

                    query_pts = np.column_stack((rows[nodata_mask], cols[nodata_mask]))

                    tree = KDTree(known_pts)

                    distances, idxs = tree.query(query_pts, k=k_use)

                    weights = 1.0 / (distances + 1e-10)
                    weights /= weights.sum(axis=1, keepdims=True)

                    filled_vals = np.sum(weights * known_vals[idxs], axis=1)

                    z_band[nodata_mask] = filled_vals

                    chunk[3] = z_band

                    print(
                        f"Chunk [{chunk_id}/{total_chunks}] y={y} x={x} → filled {nodata_count:,} pixels"
                    )

                # Extract core region (remove overlap)
                y_offset = y - y_read
                x_offset = x - x_read

                h_write = min(chunk_size, H - y)
                w_write = min(chunk_size, W - x)

                core = chunk[
                    :,
                    y_offset : y_offset + h_write,
                    x_offset : x_offset + w_write
                ]

                # Edge mask to prevent boundary artifacts
                mask = np.ones((h_write, w_write), dtype=bool)

                if y == 0:
                    mask[:overlap, :] = False

                if x == 0:
                    mask[:, :overlap] = False

                if y + chunk_size >= H:
                    mask[-overlap:, :] = False

                if x + chunk_size >= W:
                    mask[:, -overlap:] = False

                # Apply mask
                for b in range(core.shape[0]):
                    band = core[b]
                    band[~mask] = nodata
                    core[b] = band

                write_window = Window(x, y, w_write, h_write)

                dst.write(core, window=write_window)

    print("\n─── Complete ───")
    print(f"Chunks processed : {chunk_id}")
    print(f"Output saved to  : {output_tif}")
    
if __name__ == "__main__":
    fill_nodata_knn_chunkwise(input_file_path,output_file_path)