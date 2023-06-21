def get_movement(var_magnitudes, var_border_magnitudes, mean_motions_x, mean_motions_y):
    movement = ""
    if var_magnitudes is not None:
        if var_border_magnitudes is None or var_border_magnitudes < 13:
            return "STATIC"
    else:
        if abs(mean_motions_x) > 0.2:
            movement += "PAN"
        if abs(mean_motions_y) > 0.2:
            movement += ", TILT"
    return movement


def main():
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames/1"


if __name__ == "__main__":
    main()
