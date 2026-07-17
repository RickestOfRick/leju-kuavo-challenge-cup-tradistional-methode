"""Task 2 class-specific resting body heights above the table plane."""


# Shared correction from fitted table Z to the mode-0 motion reference Z.
GLOBAL_Z_CALIBRATION_OFFSET = -0.0009

# MuJoCo body-frame origin relative to the supporting table surface.
BODY_ORIGIN_Z_OFFSET_FROM_TABLE = {
    "part_type_a": -0.0272,
    "part_type_b": -0.00004,
    "part_type_c": 0.0,
}


def body_origin_z_from_table(class_name, xy, table_plane):
    coefficients = [float(value) for value in table_plane["coefficients"]]
    if len(coefficients) != 3:
        raise ValueError("table plane must contain z=ax+by+c coefficients")
    if class_name not in BODY_ORIGIN_Z_OFFSET_FROM_TABLE:
        raise ValueError("unsupported task2 part class: {}".format(class_name))
    x, y = [float(value) for value in xy[:2]]
    table_z = coefficients[0] * x + coefficients[1] * y + coefficients[2]
    return (
        table_z
        + GLOBAL_Z_CALIBRATION_OFFSET
        + BODY_ORIGIN_Z_OFFSET_FROM_TABLE[class_name]
    )
