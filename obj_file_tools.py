import sys
import numpy as np


def load_obj_file(input_file):
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            normalize = []
            if line and line[0] in ['v', 'f', 'vn']:
                split = line.strip().split()
                normalize = [split[0]] + [float(x) for x in split[1:]]

            lines.append(normalize)

    return np.array(lines)


def rotate_obj_file(input_file, output_file, angles_deg):
    vertices = []
    faces = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                face_indices = line.strip().split()[1:]
                face = []
                for idx in face_indices:
                    vertex_indices = idx.split('//')[0]
                    face.append(int(vertex_indices) - 1)
                faces.append(face)

    vertices = np.array(vertices)

    angles_rad = np.radians(angles_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                   [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])

    Ry = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                   [0, 1, 0],
                   [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])

    Rz = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                   [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                   [0, 0, 1]])

    vertices = np.dot(Rz, np.dot(Ry, np.dot(Rx, vertices.T))).T

    with open(output_file, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')

        for face in faces:
            f.write('f')
            for idx in face:
                f.write(f' {idx + 1}')
            f.write('\n')

    print(f'Rotated OBJ file saved to {output_file}')


def flip_axis(obj_file, flip_xy=False, flip_xz=False, flip_yz=False):
    vertices = []
    faces = []

    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                faces.append(line.strip())

    vertices = np.array(vertices)

    if flip_xy:
        vertices[:, [0, 1]] = vertices[:, [1, 0]]
    if flip_xz:
        vertices[:, [0, 2]] = vertices[:, [2, 0]]
    if flip_yz:
        vertices[:, [1, 2]] = vertices[:, [2, 1]]

    output_file = obj_file.replace('.obj', '_flipped.obj')
    with open(output_file, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            f.write(f"{face}\n")

    print(f"Flipped model saved to: {output_file}")


def scale_obj(obj_file, scale_factor=1.0, scale_x=1.0, scale_y=1.0):
    vertices = []
    faces = []

    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                faces.append(line.strip())

    vertices = np.array(vertices)

    vertices[:, 0] *= scale_x
    vertices[:, 1] *= scale_y

    output_file = obj_file.replace('.obj', '_scaled.obj')
    with open(output_file, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            f.write(f"{face}\n")

    print(f"Scaled model saved to: {output_file}")


def estimate_model_size(obj_file):
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                x, y, z = float(x), float(y), float(z)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_z = min(min_z, z)
                max_z = max(max_z, z)

    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z

    print(f"Estimated dimensions of {obj_file}:")
    print(f"Width: {width:.2f} units")
    print(f"Height: {height:.2f} units")
    print(f"Depth: {depth:.2f} units")


def check_coordinate_system(obj_file):
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                coords = list(map(float, line.strip().split()[1:4]))
                if len(coords) >= 3:
                    x, y, z = coords
                    if abs(z) > abs(x) and abs(z) > abs(y):
                        return 'Z-up'
                    elif abs(y) > abs(x) and abs(y) > abs(z):
                        return 'Y-up'
                    else:
                        return 'Unknown'
                    

def flip_coordinate_system(vertices, from_system='Y-up', to_system='Z-up'):
    if from_system == to_system:
        return vertices
    
    vertices = np.array(vertices)
    
    if from_system == 'Y-up' and to_system == 'Z-up':
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
    elif from_system == 'Z-up' and to_system == 'Y-up':
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
    
    return vertices.tolist()


def rotate_to_standard_orientation(obj_files):
    standard_up = np.array([0, 1, 0])
    rotated_models = []
    
    for obj_file in obj_files:
        vertices = []
        
        with open(obj_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertex = list(map(float, line.strip().split()[1:4]))
                    vertices.append(vertex)
            
        model_up = np.mean(vertices, axis=0)
        
        rotation_axis = np.cross(standard_up, model_up)
        rotation_angle = np.arccos(np.dot(standard_up, model_up) / (np.linalg.norm(standard_up) * np.linalg.norm(model_up)))
        
        def rotation_matrix(axis, theta):
            axis = axis / np.linalg.norm(axis)
            a = np.cos(theta / 2.0)
            b, c, d = -axis * np.sin(theta / 2.0)
            return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                             [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                             [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])
        
        rotation_mat = rotation_matrix(rotation_axis, rotation_angle)
        rotated_vertices = np.dot(vertices, rotation_mat)
        rotated_models.append(rotated_vertices)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python rotate_obj.py input.obj output_rotated.obj x_angle y_angle z_angle")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    angles_deg = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
