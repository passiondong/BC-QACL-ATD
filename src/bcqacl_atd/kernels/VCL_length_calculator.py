from itertools import product


def calculate_l(w: float, ratio_l_w: float) -> float:
    return w * ratio_l_w


def calculate_compensation_msl(
    w: float,
    ratio_l_w: float,
    w_line_um: float | None = None,
) -> dict[str, float]:
    """Return the compensation microstrip dimensions for one W/R corner.

    The formulas intentionally mirror the local geometry calculator:
      W_MSL = w_line_um when supplied, otherwise 0.25 * W
      L_MSL = Compensate_Lline / 3

    Units follow the input geometry, normally um.
    """
    l = calculate_l(w, ratio_l_w)
    distance_gnd = 5

    w_line = float(w_line_um) if w_line_um is not None else 0.25 * w
    chamfer_side_length = 0.569 * w_line
    w_port = 0.64 * w_line
    distance_port = (w - chamfer_side_length * 2 - w_port * 3) / 4

    length_tap_out = (w - w_port) / 2
    length_tap_in = (w - 2 * w_line) / 2
    length_tap_middle = (length_tap_out + length_tap_in) / 2

    length_port_in = w - 2 * w_line - w_port - 2 * distance_port
    length_port_out = (w - 3 * w_port - 2 * distance_port) / 2
    length_port_middle = (length_port_in + length_port_out) / 2

    length_long = l + distance_gnd * 2
    length_straight_line = length_long + length_tap_middle + length_port_middle
    compensate_lline = 0.5 * w_port + distance_port
    compensate_lline2 = 0.75 * w_port + distance_port
    l_compensate_line = compensate_lline / 3

    return {
        "L_um": l,
        "W_MSL_um": w_line,
        "L_MSL_um": l_compensate_line,
        "chamfer_side_length_um": chamfer_side_length,
        "w_port_um": w_port,
        "distance_port_um": distance_port,
        "Compensate_Lline_um": compensate_lline,
        "Compensate_Lline2_um": compensate_lline2,
        "real_coupling_length1_um": length_straight_line - compensate_lline2,
    }


def calculate_real_coupling_length1(
    w: float,
    ratio_l_w: float,
    w_line_um: float | None = None,
) -> float:
    return calculate_compensation_msl(w, ratio_l_w, w_line_um=w_line_um)["real_coupling_length1_um"]


def generate_results(
    w_values: list[float],
    ratio_values: list[float],
    w_line_um: float | None = None,
) -> list[tuple[float, float, float, float, float, float]]:
    results = []
    for w, ratio_l_w in product(w_values, ratio_values):
        l = calculate_l(w, ratio_l_w)
        compensation = calculate_compensation_msl(w, ratio_l_w, w_line_um=w_line_um)
        real_coupling_length1 = compensation["real_coupling_length1_um"]
        w_line = compensation["W_MSL_um"]
        l_compensate_line = compensation["L_MSL_um"]
        results.append((w, ratio_l_w, l, real_coupling_length1, w_line, l_compensate_line))
    return results


if __name__ == "__main__":
    w_values = [90, 100,101, 106, 110, 120]
    ratio_values = [1.3, 1.4, 1.5, 1.6, 1.65, 1.7]

    results = generate_results(w_values, ratio_values)

    print(f"{'W':>7} {'R':>7} {'L':>8} {'real_coupling_length1':>24} {'w_line':>10} {'l_compensate_line':>20}")
    print("-" * 82)
    for w, ratio_l_w, l, real_coupling_length1, w_line, l_compensate_line in results:
        print(f"{w:7.2f} {ratio_l_w:7.2f} {l:8.1f} {real_coupling_length1:24.1f} {w_line:10.2f} {l_compensate_line:20.2f}")
