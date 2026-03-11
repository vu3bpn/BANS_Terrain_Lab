import laspy
import itertools
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import pandas
from config import *
from keystore import *
from misc_utilities import *


if __name__ == "__main1__":    
    #def convert_to_tiles(subset_n=10):
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
            if os.path.exists(out_file_path):
                continue
            las_file_p = laspy.open(las_file)
            for points in las_file_p.chunk_iterator(chunk_size):
                x,y = points.x.copy(),points.y.copy()
                x_min1 = x_min + x_0*x_window
                x_max1 = x_min1+x_window
                y_min1 = y_min + y_0*y_window
                y_max1 = y_min1 + y_window 
                mask = (x >= x_min1) & (x < x_max1) & (y >= y_min1) & (y < y_max1)
                if sum(mask) >0 :
                    selected_points.append(points[mask].copy())
            print(las_file_p.header.point_format)
            out_header = laspy.LasHeader(point_format=las_file_p.header.point_format,version="1.4")
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
    data_info_df = pandas.read_csv(data_info_file)
    filename_paths = {os.path.split(x)[-1]:x for x in data_info_df['filename']}
    for las_file in las_vect_dict:
        vect_file = las_vect_dict[las_file]
        shapefile_path = os.path.join(vector_dir,vect_file)
        las_file_path = filename_paths[las_file]
        #subset_with_shapefile(las_file_path,shapefile_path)
        gdf = gpd.read_file(shapefile_path)
        records = []
        headers = []
        for row1 in gdf.iterrows():
            idx   =  row1[1]['id']
            record1,header1 = subset_with_geom(las_file_path,row1[1].geometry)
            records.append(record1)
            headers.append(header1)






#def subset_with_shapefile(las_file_path,shapefile_path):
if __name__ == "__main1__":
    gdf = gpd.read_file(shapefile_path)
    las_file_p = laspy.open(las_file_path)
    #bounds = list(gdf.geometry.bounds)
    #ids = gdf['id']    
    variables = list(las_file_p.header.point_format.dimension_names)
    #for idx,bound in zip(ids,bounds):


    
    for row1 in gdf.iterrows():
        #bound = row1[1].geometry.bounds
        idx   =  row1[1]['id']
        (x_min,
        y_min,
        x_max,        
        y_max) = row1[1].geometry.bounds
        selected_points = []
        out_filename = os.path.split(las_file_path)[-1].split(".")[0]+f"_id_{idx}.las"            
        out_file_path = os.path.join(debug_subset_dir,out_filename)  
        if os.path.exists(out_file_path):
            continue
        las_file_p = laspy.open(las_file_path)
        for points in las_file_p.chunk_iterator(chunk_size):
            x,y = points.x.copy(),points.y.copy()            
            mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
            if sum(mask) >0 :
                selected_points.append(points[mask].copy())            
            
        print(las_file_p.header.point_format)
        out_header = laspy.LasHeader(point_format=las_file_p.header.point_format,version="1.4")
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
            with laspy.open(out_file_path, mode='w', header=out_header) as writer:
                writer.write_points(record)
            
            '''
            with laspy.open(out_file_path,mode='w',header = out_header) as writer:
                for points in selected_points:
                    writer.write_points(points)
                log(f"generated {out_file_path}")  
            '''




    

    
if __name__ == "__main__":
    generate_data_info()
    




    


    
    
if __name__ == "__main__":
    '''subset each scene into 5x5 sub scenes'''
    las_input_filenames = list(Path(input_laz_dir).rglob("*.las"))
    for las_file in las_input_filenames:  
        split_into(las_file,10)


if __name__ == "__main1__":
    '''subset into 10x10 windows'''
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz"))
    for las_file in las_input_filenames:  
        print(las_file)
        las_file_p = laspy.open(las_file)        
        x_min = las_file_p.header.min[0] # Assuming x is 0 and y is 1 always
        x_max = las_file_p.header.max[0]
        y_min = las_file_p.header.min[1]
        y_max = las_file_p.header.max[1]
        window = 10
        idx = 1
        for x_0,y_0 in tqdm(list(itertools.product(np.arange(x_min,x_max,window),np.arange(y_min,y_max,window)))):
            out_filename = os.path.split(las_file)[-1].split('.')[0]+f"subset_{idx}.laz"
            idx+=1
            out_file_path =  os.path.join(fixed_window_subset_dir,out_filename)
            if not os.path.exists(out_file_path):                
                record,out_header = subset_las_record(las_file,x_0,y_0,window)
                if record is None:
                    continue
                with laspy.open(out_file_path, mode='w', header=out_header) as writer:
                    writer.write_points(record)
            
    

if __name__ == "__main1__":
    #convert_to_tiles()
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
        window = 100
        center_x = (x_max+x_min)/2
        center_y = (y_max+y_min)/2


        out_filename = os.path.split(las_file)[-1].split('.')[0]+f"subset_1.laz"
        selected_points = []
        out_file_path =  os.path.join(fixed_window_subset_dir,out_filename)        
        las_file_p = laspy.open(las_file)
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
            with laspy.open(out_file_path, mode='w', header=out_header) as writer:
                writer.write_points(record)
            log(f"generated {out_file_path}")
            