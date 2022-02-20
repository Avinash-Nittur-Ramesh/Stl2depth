import cv2
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Preview model
def preview_model(model):
    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    collection = mplot3d.art3d.Poly3DCollection(model.vectors)
    collection.set_edgecolor((0,0,0))
    # Add the vectors to the plot
    axes.add_collection3d(collection)
    # Auto scale to the mesh size
    scale = model.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    # Show the plot to the screen
    pyplot.show()
    
# Determine if the normal at pixel location is oriented with mesh triangle normal
def oriented(p,v0,v1,v2):
    n1 = np.cross(v2-v1, p-v1)
    n2 = np.cross(v2-v1, v0-v1)
    if np.dot(n1,n2) >= 0:
        return True
    else:
        return False

# Determine if a pixel is inside or on the mesh triangle
def pixel_in_triangle(p,v0,v1,v2):
    if oriented(p,v0,v1,v2) and oriented(p,v1,v0,v2) and oriented(p,v2,v0,v1):
        return True
    else:
        return False

# Calculate depth of valid pixels locations
def calculate_depth(x,y,index):
    # Get the normals
    nx, ny, nz = model.normals[index]
    # Get any one vertex
    v0 = model.v0[index]
    v1 = model.v1[index]
    v2 = model.v2[index]
    
    # Calculate depth at (x,y) from equation of a plane in 3D resulting in linear interpolation
    depth = 0
    p = np.array([x,y])
    if pixel_in_triangle(p,v0[:2],v1[:2],v2[:2]):
        if nz != 0:
            depth = ((nx*(x-v0[0]) + ny*(y-v0[1]))/(-nz)) + v0[2]
    return depth


if __name__ == '__main__':

    DEBUG = True
    DISPLAYPLOT = True
    SAVEIMG = True
    DISPLAYIMG = True
    
    width = 200
    
    path = "./"
    # file_name = "modified_cube_ascii"
    # file_name = "Dolphin_cropped"
    file_name = "tactile_display_1_basic"
    # file_name = "crater"
    file_ext = ".stl"
    file_path = path + file_name + file_ext;
    
    # Load the STL file
    model = mesh.Mesh.from_file(file_path)
    
    # Theta is in radians, and it rotates in counter-clockwise direction, so use -ve theta for clockwise rotation
    # Specify the axes of rotation as unit vector in [x,y,z]
    ## model.rotate([1,0,0],theta=np.deg2rad(-90))
    
    # Debugging
    if DEBUG:
        print("mesh.shape: ", model.points.shape)
        print("z mean: ", model.z.mean())
        print("z.max: ", model.z.max())

    # Preview model
    if DISPLAYPLOT:
        preview_model(model)
    
    # Translate the model to positive coordinates
    model.translate(np.array([-model.x.min(),-model.y.min(),-model.z.min()]))

    # Calculate height of image while maintaining aspect ratio
    height = (model.y.max() - model.y.min()) * width / (model.x.max() - model.x.min());
    height = int(np.fabs(np.ceil(height)));

    # Scale the model while maintaining aspect ratios or surface normals according to the width
    scale = width / model.x.max()
    model.x = model.x * scale
    model.y = model.y * scale
    model.z = model.z * scale

    # Create a depth image
    im = np.zeros((height+1, width+1))
    for triangle_index in range(model.normals.shape[0]):
        for y in range(int(model.y[triangle_index].min()), int(model.y[triangle_index].max())+1):
            for x in range(int(model.x[triangle_index].min()), int(model.x[triangle_index].max())+1):
                depth = calculate_depth(x,y,triangle_index)
                if im[y,x] < depth:
                    im[y,x] = depth
    
    # Adjust the dynamic brightness range of the depth image    
    im = (im-im.min())*255 / (im.max()-im.min())

    # Visualise the final output
    if DISPLAYIMG:
        cv2.imshow("Output", im[::-1,:].astype(np.uint8))
        print("\n\nshape: ", im.shape)

        cv2.waitKey(0)

    if SAVEIMG:
        save_path = str(file_name + ".png")
        print("saving at this path: ", save_path)
        cv2.imwrite(save_path, im)