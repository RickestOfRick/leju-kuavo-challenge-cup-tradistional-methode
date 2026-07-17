"""Load task visual mesh vertices in their MuJoCo body frames."""

import os
import struct
import xml.etree.ElementTree as ET


def parse_vector(text, length, default):
    if not text:
        return list(default)
    values = [float(value) for value in str(text).replace(",", " ").split()]
    if len(values) != length:
        raise ValueError("expected {} values, got {}".format(length, len(values)))
    return values


def rotate_vector_wxyz(quaternion, vector):
    w, x, y, z = [float(value) for value in quaternion]
    vx, vy, vz = [float(value) for value in vector]
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm <= 1e-12:
        raise ValueError("mesh quaternion norm is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def load_obj_vertices(path):
    vertices = []
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) >= 4 and fields[0] == "v":
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return vertices


def load_obj_triangles(path):
    vertices = []
    triangles = []
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) >= 4 and fields[0] == "v":
                vertices.append(
                    [float(fields[1]), float(fields[2]), float(fields[3])]
                )
                continue
            if len(fields) < 4 or fields[0] != "f":
                continue
            indices = []
            for field in fields[1:]:
                index = int(field.split("/", 1)[0])
                index = index - 1 if index > 0 else len(vertices) + index
                if index < 0 or index >= len(vertices):
                    raise ValueError("OBJ face index is outside vertex array: {}".format(path))
                indices.append(index)
            for offset in range(1, len(indices) - 1):
                triangles.append(
                    [
                        vertices[indices[0]],
                        vertices[indices[offset]],
                        vertices[indices[offset + 1]],
                    ]
                )
    return triangles


def load_stl_vertices(path):
    with open(path, "rb") as stream:
        raw = stream.read()
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        expected_size = 84 + 50 * triangle_count
        if triangle_count > 0 and expected_size == len(raw):
            vertices = []
            for triangle_index in range(triangle_count):
                offset = 84 + 50 * triangle_index + 12
                values = struct.unpack_from("<9f", raw, offset)
                vertices.extend(
                    [list(values[0:3]), list(values[3:6]), list(values[6:9])]
                )
            return vertices

    vertices = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return vertices


def load_stl_triangles(path):
    vertices = load_stl_vertices(path)
    if len(vertices) % 3 != 0:
        raise ValueError("STL vertex count is not divisible by three: {}".format(path))
    return [vertices[index : index + 3] for index in range(0, len(vertices), 3)]


def _load_scaled_vertices(path, scale, cache):
    key = (os.path.abspath(path), tuple(float(value) for value in scale))
    if key in cache:
        return cache[key]
    extension = os.path.splitext(path)[1].lower()
    if extension == ".obj":
        vertices = load_obj_vertices(path)
    elif extension == ".stl":
        vertices = load_stl_vertices(path)
    else:
        raise ValueError("unsupported visual mesh format: {}".format(extension))
    scaled = [
        [vertex[axis] * scale[axis] for axis in range(3)]
        for vertex in vertices
    ]
    if not scaled:
        raise ValueError("visual mesh contains no vertices: {}".format(path))
    cache[key] = scaled
    return scaled


def _load_scaled_triangles(path, scale, cache):
    key = (os.path.abspath(path), tuple(float(value) for value in scale))
    if key in cache:
        return cache[key]
    extension = os.path.splitext(path)[1].lower()
    if extension == ".obj":
        triangles = load_obj_triangles(path)
    elif extension == ".stl":
        triangles = load_stl_triangles(path)
    else:
        raise ValueError("unsupported visual mesh format: {}".format(extension))
    scaled = [
        [
            [vertex[axis] * scale[axis] for axis in range(3)]
            for vertex in triangle
        ]
        for triangle in triangles
    ]
    if not scaled:
        raise ValueError("visual mesh contains no triangles: {}".format(path))
    cache[key] = scaled
    return scaled


def load_body_visual_vertices(model_path, required_body_names):
    """Return raw visual vertices transformed into each requested body frame."""
    root = ET.parse(model_path).getroot()
    xml_directory = os.path.dirname(os.path.abspath(model_path))
    required = {str(name) for name in required_body_names}

    mesh_assets = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = str(mesh.get("name", ""))
            mesh_file = str(mesh.get("file", ""))
            if not name or not mesh_file:
                continue
            mesh_assets[name] = {
                "path": os.path.normpath(os.path.join(xml_directory, mesh_file)),
                "scale": parse_vector(mesh.get("scale"), 3, [1.0, 1.0, 1.0]),
            }

    cache = {}
    by_body = {}
    for body in root.findall(".//body"):
        body_name = str(body.get("name", ""))
        if body_name not in required:
            continue
        body_vertices = []
        for geom in body.findall("geom"):
            mesh_name = str(geom.get("mesh", ""))
            if not mesh_name or mesh_name not in mesh_assets:
                continue
            if geom.get("euler") or geom.get("axisangle") or geom.get("xyaxes"):
                raise ValueError(
                    "visual mesh geom {} uses an unsupported local rotation".format(
                        str(geom.get("name", mesh_name))
                    )
                )
            asset_spec = mesh_assets[mesh_name]
            mesh_vertices = _load_scaled_vertices(
                asset_spec["path"], asset_spec["scale"], cache
            )
            local_position = parse_vector(geom.get("pos"), 3, [0.0, 0.0, 0.0])
            local_quaternion = parse_vector(
                geom.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]
            )
            for vertex in mesh_vertices:
                rotated = rotate_vector_wxyz(local_quaternion, vertex)
                body_vertices.append(
                    [rotated[axis] + local_position[axis] for axis in range(3)]
                )
        if body_vertices:
            by_body[body_name] = body_vertices
    return by_body


def load_body_visual_triangles(model_path, required_body_names):
    """Return visual triangles transformed into each requested MuJoCo body frame."""
    root = ET.parse(model_path).getroot()
    xml_directory = os.path.dirname(os.path.abspath(model_path))
    required = {str(name) for name in required_body_names}

    mesh_assets = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = str(mesh.get("name", ""))
            mesh_file = str(mesh.get("file", ""))
            if not name or not mesh_file:
                continue
            mesh_assets[name] = {
                "path": os.path.normpath(os.path.join(xml_directory, mesh_file)),
                "scale": parse_vector(mesh.get("scale"), 3, [1.0, 1.0, 1.0]),
            }

    cache = {}
    by_body = {}
    for body in root.findall(".//body"):
        body_name = str(body.get("name", ""))
        if body_name not in required:
            continue
        body_triangles = []
        for geom in body.findall("geom"):
            mesh_name = str(geom.get("mesh", ""))
            if not mesh_name or mesh_name not in mesh_assets:
                continue
            if geom.get("euler") or geom.get("axisangle") or geom.get("xyaxes"):
                raise ValueError(
                    "visual mesh geom {} uses an unsupported local rotation".format(
                        str(geom.get("name", mesh_name))
                    )
                )
            asset_spec = mesh_assets[mesh_name]
            mesh_triangles = _load_scaled_triangles(
                asset_spec["path"], asset_spec["scale"], cache
            )
            local_position = parse_vector(geom.get("pos"), 3, [0.0, 0.0, 0.0])
            local_quaternion = parse_vector(
                geom.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]
            )
            for triangle in mesh_triangles:
                transformed = []
                for vertex in triangle:
                    rotated = rotate_vector_wxyz(local_quaternion, vertex)
                    transformed.append(
                        [
                            rotated[axis] + local_position[axis]
                            for axis in range(3)
                        ]
                    )
                body_triangles.append(transformed)
        if body_triangles:
            by_body[body_name] = body_triangles
    return by_body


def load_body_visual_triangles_by_geom(model_path, required_body_names):
    """Return visual triangles grouped by body and visual geom name."""
    root = ET.parse(model_path).getroot()
    xml_directory = os.path.dirname(os.path.abspath(model_path))
    required = {str(name) for name in required_body_names}

    mesh_assets = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = str(mesh.get("name", ""))
            mesh_file = str(mesh.get("file", ""))
            if not name or not mesh_file:
                continue
            mesh_assets[name] = {
                "path": os.path.normpath(os.path.join(xml_directory, mesh_file)),
                "scale": parse_vector(mesh.get("scale"), 3, [1.0, 1.0, 1.0]),
            }

    cache = {}
    by_body = {}
    for body in root.findall(".//body"):
        body_name = str(body.get("name", ""))
        if body_name not in required:
            continue
        by_geom = {}
        for geom in body.findall("geom"):
            mesh_name = str(geom.get("mesh", ""))
            if not mesh_name or mesh_name not in mesh_assets:
                continue
            if geom.get("euler") or geom.get("axisangle") or geom.get("xyaxes"):
                raise ValueError(
                    "visual mesh geom {} uses an unsupported local rotation".format(
                        str(geom.get("name", mesh_name))
                    )
                )
            asset_spec = mesh_assets[mesh_name]
            mesh_triangles = _load_scaled_triangles(
                asset_spec["path"], asset_spec["scale"], cache
            )
            local_position = parse_vector(geom.get("pos"), 3, [0.0, 0.0, 0.0])
            local_quaternion = parse_vector(
                geom.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]
            )
            transformed_triangles = []
            for triangle in mesh_triangles:
                transformed = []
                for vertex in triangle:
                    rotated = rotate_vector_wxyz(local_quaternion, vertex)
                    transformed.append(
                        [
                            rotated[axis] + local_position[axis]
                            for axis in range(3)
                        ]
                    )
                transformed_triangles.append(transformed)
            geom_name = str(geom.get("name", mesh_name))
            by_geom[geom_name] = transformed_triangles
        if by_geom:
            by_body[body_name] = by_geom
    return by_body
