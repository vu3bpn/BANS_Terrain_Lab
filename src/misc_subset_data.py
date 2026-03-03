import laspy
import itertools
import numpy as np
from config import *

if __name__ == "__main__":
    las_input_filenames = list(Path(copc_dir).rglob("*.laz"))[:1]  
    for las_file in las_input_filenames:    
        print(las_file)
        las_file_p = laspy.open(las_file)
        
        '''
        x_min_list =[]
        x_max_list = []
        y_min_list = []
        y_max_list  = []
        for i, points in enumerate(las_file_p.chunk_iterator(chunk_size)):
            x_list = np.array(points["X"])
            y_list = np.array(points["X"])
            x_min_list.append(min(x_list))
            x_max_list.append(max(x_list))
            y_min_list.append(min(y_list))
            y_max_list.append(max(y_list))
        '''

        x_min = las_file_p.header.min[0] # Assuming x is 0 and y is 1 always
        x_max = las_file_p.header.max[0]
        y_min = las_file_p.header.min[1]
        y_max = las_file_p.header.max[1]
        
        
        variables = list(las_file_p.header.point_format.dimension_names)
        
        log(f"X range : {x_min} - {x_max} ,Y range : {y_min} - {y_max}") 
        
        x_window = int((x_max-x_min)/subset_n)+1
        y_window = int((y_max-y_min)/subset_n)+1
        selected_points = []
        
        for x_0,y_0 in itertools.product(range(subset_n),range(subset_n)):
            out_filename = os.path.split(las_file)[-1].strip(".laz")+f"subset_{x_0}_{y_0}.laz"
            out_file_path = os.path.join(debug_subset_dir,out_filename)          
            with laspy.open(out_file_path,mode='w',header = las_file_p.header) as writer:
                for points in las_file_p.chunk_iterator(chunk_size):
                    x,y = points.x.copy(),points.y.copy()
                    mask = (x >= x_min+ x_0*x_window) & (x < (x_min+(x_0+1)*x_window)) & \
                            (y >= y_min + y_0*y_window) & (y < (y_min+(y_0+1)*y_window))
                    #selected_points.extend(points[mask].copy())
                    #print(len(points))
                    writer.append_points(points[mask])
                
            #output_file = laspy.LasData(las_file_p.header)
            #output_file.points = selected_points
            #output_file.write(out_file_path)