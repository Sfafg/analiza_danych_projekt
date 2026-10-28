import utils

folder = utils.prepare_build_dir("charakterystyka")
data = utils.get_data()

with open(f"{folder}/charakterystyka.txt", "w") as f:
    for column in data.columns:
        if column == "appid":
            continue

        column_data = data[column]

        min = None
        max = None
        mean = None
        median = None
        zero_values = None
        if column_data.dtype != "object":
            min = column_data.min()
            max = column_data.max()
            mean = column_data.mean()
            median = column_data.median()
            zero_values = len(column_data[column_data == 0])

        print(column, file=f)
        print("Minimum:", min, file=f)
        print("Maximum:", max, file=f)
        print("Średnia:", mean, file=f)
        print("Mediana:", median, file=f)
        print("Ilość wartości zerowych:", zero_values, file=f)
        print("", file=f)
