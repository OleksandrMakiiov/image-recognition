def is_inside(outside_box, inside_box):
    x_min_out, y_min_out, x_max_out, y_max_out = outside_box
    x_min_in, y_min_in, x_max_in, y_max_in = inside_box

    return (x_min_out <= x_min_in and
            y_min_out <= y_min_in and
            x_max_out >= x_max_in and
            y_max_out >= y_max_in)


outside_box = [0.43193933367729187, 0.39147400856018066, 0.4944233000278473, 0.4196205139160156]
potentially_inside_box = [0.4410521388053894, 0.4004344642162323, 0.48614373803138733, 0.41346850991249084]

obj_box = {list: 4} [0.4974689185619354, 0.39144542813301086, 0.5671273469924927, 0.41987907886505127]
ocr_box = {list: 4} [0.5058712959289551, 0.40188270807266235, 0.5171442031860352, 0.4120202660560608]

print(is_inside(outside_box, potentially_inside_box))  # False
