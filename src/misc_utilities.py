import laspy
import itertools
import numpy as np
import geopandas as gpd
import pandas
from tqdm import tqdm
from config import *

def generate_data_info():
    las_input_filenames = list(Path(input_laz_dir).rglob("*.las"))
    data_df = []
    for filename1 in las_input_filenames:
        las_file_p = laspy.open(filename1)
        x_min = las_file_p.header.min[0] # Assuming x is 0 and y is 1 always
        x_max = las_file_p.header.max[0]
        y_min = las_file_p.header.min[1]
        y_max = las_file_p.header.max[1]
        size = las_file_p.header.point_count
        crs = las_file_p.header.parse_crs().to_epsg()
        variables = list(las_file_p.header.point_format.dimension_names)  
        data_df.append({"filename":filename1,
                        "size":size,
                        "x_min":x_min,
                        "x_max":x_max,
                        "y_min":y_min,
                        "y_max":y_max,
                        "crs":crs,
                        "variables":variables})
    df = pandas.DataFrame.from_dict(data_df)
    df.to_csv(data_info_file,index= False)

def subset_las_record(las_file_path,center_x,center_y,window):
    las_file_p = laspy.open(las_file_path)
    variables = list(las_file_p.header.point_format.dimension_names)  
    selected_points = []
    for points in las_file_p.chunk_iterator(chunk_size):
        x,y = points.x.copy(),points.y.copy()
        x_min1 = center_x 
        x_max1 = center_x + window
        y_min1 = center_y 
        y_max1 = center_y + window
        mask = (x >= x_min1) & (x < x_max1) & (y >= y_min1) & (y < y_max1)
        if sum(mask) >0 :
            selected_points.append(points[mask].copy())         
        out_header = laspy.LasHeader(point_format=las_file_p.header.point_format,
                                     version=las_file_p.header.version)
        out_header.x_scale = las_file_p.header.x_scale
        out_header.y_scale = las_file_p.header.y_scale
        out_header.z_scale = las_file_p.header.z_scale
        out_header.offsets = las_file_p.header.offsets            
        out_crs = las_file_p.header.parse_crs()
        out_header.add_crs(out_crs)
    if len(selected_points) >0:
        total_points = sum(len(x) for x in selected_points)
        record = laspy.ScaleAwarePointRecord.zeros(total_points, header=out_header)
        for var in variables:
            start_idx = 0
            for points in selected_points:
                end_idx = start_idx + len(points)
                record[var][start_idx:end_idx] = points[var]
                start_idx = end_idx
        return record,out_header
    return None,None

def split_into(las_file_path,n):
    las_file_p = laspy.open(las_file_path)        
    x_min = las_file_p.header.min[0] # Assuming x is 0 and y is 1 always
    x_max = las_file_p.header.max[0]
    y_min = las_file_p.header.min[1]
    y_max = las_file_p.header.max[1]
    window = max((x_max-x_min)/n,(y_max-y_min)/n)
    log(f"window size {window} for {las_file_path}")
    idx = 1
    for x_0,y_0 in tqdm(list(itertools.product(np.arange(x_min,x_max,window),np.arange(y_min,y_max,window)))):
        out_filename = os.path.split(las_file_path)[-1].split('.')[0]+f"_subset_{idx}.las"
        idx+=1
        out_file_path =  os.path.join(split_files_subset_dir,out_filename)
        if not os.path.exists(out_file_path):                
            record,out_header = subset_las_record(las_file_path,x_0,y_0,window)
            if record is None:
                continue
            with laspy.open(out_file_path, mode='w', header=out_header) as writer:
                writer.write_points(record)

def subset_with_geom(las_file_path,geom):
    las_file_p = laspy.open(las_file_path)
    variables = list(las_file_p.header.point_format.dimension_names)  
    selected_points = []
    for points in las_file_p.chunk_iterator(chunk_size):
        x,y = points.x.copy(),points.y.copy()        
        mask = geom.contains(gpd.points_from_xy(x,y))
        if sum(mask) >0 :
            selected_points.append(points[mask].copy())         
        out_header = laspy.LasHeader(point_format=las_file_p.header.point_format,
                                     version=las_file_p.header.version)
        out_header.x_scale = las_file_p.header.x_scale
        out_header.y_scale = las_file_p.header.y_scale
        out_header.z_scale = las_file_p.header.z_scale
        out_header.offsets = las_file_p.header.offsets            
        out_crs = las_file_p.header.parse_crs()
        out_header.add_crs(out_crs)
    if len(selected_points) >0:
        total_points = sum(len(x) for x in selected_points)
        record = laspy.ScaleAwarePointRecord.zeros(total_points, header=out_header)
        for var in variables:
            start_idx = 0
            for points in selected_points:
                end_idx = start_idx + len(points)
                record[var][start_idx:end_idx] = points[var]
                start_idx = end_idx
        return record,out_header
    return None,None