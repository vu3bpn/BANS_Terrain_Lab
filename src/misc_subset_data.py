import laspy
import itertools
import numpy as np
from config import *


    
def convert_to_tiles(subset_n=10):
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz")) 
    for las_file in las_input_filenames:    
        print(las_file)
        las_file_p = laspy.open(las_file)        
        x_min = las_file_p.header.min[0] # Assuming x is 0 and y is 1 always
        x_max = las_file_p.header.max[0]
        y_min = las_file_p.header.min[1]
        y_max = las_file_p.header.max[1]
                
        variables = list(las_file_p.header.point_format.dimension_names)        
        log(f"X range : {x_min} - {x_max} ,Y range : {y_min} - {y_max}") 
        
        x_window = int((x_max-x_min)/subset_n)+1
        y_window = int((y_max-y_min)/subset_n)+1
        
        for x_0,y_0 in itertools.product(range(subset_n),range(subset_n)):
            selected_points = []
            out_filename = os.path.split(las_file)[-1].split(".")[0]+f"_subset_{x_0}_{y_0}.laz"
            
            out_file_path = os.path.join(debug_subset_dir,out_filename)  
            las_file_p = laspy.open(las_file)
            for points in las_file_p.chunk_iterator(chunk_size):
                x,y = points.x.copy(),points.y.copy()
                x_min1 = x_min + x_0*x_window
                x_max1 = x_min1+x_window
                y_min1 = y_min + y_0*y_window
                y_max1 = y_min1 + y_window                
                #print(x_min1,x_max1,y_min1,y_max1)
                mask = (x >= x_min1) & (x < x_max1) & (y >= y_min1) & (y < y_max1)
                if sum(mask) >0 :
                    selected_points.append(points[mask].copy())            
            
            out_header = laspy.LasHeader(point_format=7,version="1.4")
            out_header.x_scale = las_file_p.header.x_scale
            out_header.y_scale = las_file_p.header.y_scale
            out_header.z_scale = las_file_p.header.z_scale
            out_header.offsets = las_file_p.header.offsets
            
            out_crs = las_file_p.header.parse_crs()
            out_header.add_crs(out_crs)
            if len(selected_points) >0:
                with laspy.open(out_file_path,mode='w',header = out_header) as writer:
                    for points in selected_points:
                        writer.write_points(points)
                    log(f"generated {out_file_path}")
            
if __name__ == "__main__":
    convert_to_tiles()
             