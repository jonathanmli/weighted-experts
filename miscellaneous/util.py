def variables_to_dict(names):
    out = {}
    for name in names:
        out[name] = eval(name)