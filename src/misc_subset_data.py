import laspy
import itertools
from config import *

if __name__ == "__main__":
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz"))[:1]  
    for las_file in las_input_filenames:    
        print(las_file)
        las_file_p = laspy.open(las_file)
        x_min_list =[]
        x_max_list = []
        y_min_list = []
        y_max_list  = []
        for i, points in enumerate(las_file_p.chunk_iterator(chunk_size)):
            x_list = list(points["X"])
            y_list = list(points["X"])
            x_min_list.append(min(x_list))
            x_max_list.append(max(x_list))
            y_min_list.append(min(y_list))
            y_max_list.append(max(y_list))

        x_min = min(x_min_list)
        x_max = max(x_max_list)
        y_min = min(y_min_list)
        y_max = max(y_max_list)
        variables = list(las_file_p.header.point_format.dimension_names)
        log(f"X range : {x_min} - {x_max} ,Y range : {y_min} - {y_max}") 
        x_window = int((x_max-x_min)/subset_n)+1
        y_window = int((y_max-y_min)/subset_n)+1
        selected_points = []
        for x_0,y_0 in itertools.product(range(subset_n),range(subset_n)):
            out_filename = os.path.split(las_file)[-1].strip(".laz")+f"subset_{x_0}_{y_0}.laz"
            out_file_path = os.path.join(debug_subset_dir,out_filename)
            #dest_las_file = laspy.open(out_file_path,mode='w',header=las_file_p.header)
            
            for points in las_file_p.chunk_iterator(chunk_size):
                #points_dict = {x:list(points[x]) for x in variables}
                #points_df = pandas.DataFrame(points_dict)
                #points_suset = points.query(f"X >= {x_min+x_0*x_window} ")
                x,y = points.x.copy(),points.y.copy()
                mask = (x >= x_min+ x_0*x_window) & (x < (x_min+(x_0+1)*x_window)) & \
                        (y >= y_min + y_0*y_window) & (y < (y_min+(y_0+1)*y_window))
                selected_points.extend(points[mask].copy())
            #dest_las_file.write_points()
            #dest_las_file = laspy.LasData(las_file_p.header)
            #dest_las_file.write(out_file_path)
            #log(f"generated file {out_file_path}")

