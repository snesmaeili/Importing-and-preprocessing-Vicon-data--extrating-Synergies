def muscle_no(name):

    # Define muscle names and corresponding numbers
    all_names = ['TA', 'PL', 'SOL', 'MGAS', 'GC_M',
                 'RF', 'SEMT', 'BFLH', 'VL', 'VM', 'GMED', 'G_M', 'GMAX', 'GLUT_MAX',
                 'IP', 'IPS', 'ADD', 'TFL',
                 'IC', 'LG', 'RG', 'RA', 'RECT_AB', 'EO']
    all_no = [1, 2, 3, 4, 4,
              5, 6, 7, 8, 9, 10, 10, 11, 11,
              12, 12, 13, 14,
              15, 16, 16, 17, 17, 18]

    # Create a dictionary for muscle names and numbers
    muscle_dict = dict(zip(all_names, all_no))

    # Return the muscle number or 0 if not found
    return muscle_dict.get(name, 0)
