import sys
import numpy as np

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


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python rotate_obj.py input.obj output_rotated.obj x_angle y_angle z_angle")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    angles_deg = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]

    rotate_obj_file(input_file, output_file, angles_deg)
    print(check_coordinate_system(input_file))
