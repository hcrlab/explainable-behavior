from argparse import ArgumentParser
import os
import csv
import shutil


def main(): 
    parser = ArgumentParser(description="Move images to folders named for corresponding action")
    parser.add_argument("-v", "--verbose", action="store_true", help="print extra messages (default: False")
    args = parser.parse_args()
    verbose = args.verbose

    os.chdir("data/")
    while True:
        try:
            os.chdir("categorized")
            break
        except FileNotFoundError:
            os.mkdir("categorized")
    # make sure a directory exists for each label
    existing_subds = os.listdir()
    possible_labels = ["forward", "forward left", "forward right", "backward",
        "backward left", "backward right", "left", "right", "stop", "none"]
    for possible_label in possible_labels:
        if possible_label not in existing_subds:
            os.mkdir(possible_label)

    # get list of raw data directories
    dirs = os.listdir("../all data")

    # for each directory, open data.csv
    for d in dirs:
        print("processing {} images".format(d))
        with open("../all data/{}/data.csv".format(d)) as data:
            data_reader = csv.reader(data)
            # get header
            header = next(data_reader)
            up_idx = header.index("wheels: up key")
            down_idx = header.index("wheels: down key")
            left_idx = header.index("wheels: left key")
            right_idx = header.index("wheels: right key")
            stop_idx = header.index("stop")
            row_number = 1
            for row in data_reader:
                forward = int(row[up_idx])
                backward = int(row[down_idx])
                left = int(row[left_idx])
                right = int(row[right_idx])
                stop = int(row[stop_idx])
                labels = []
                if forward:
                    labels.append("forward")
                if backward:
                    labels.append("backward")
                if left:
                    labels.append("left")
                if right:
                    labels.append("right")
                # stop overrides all other labels
                if stop:
                    labels = ["stop"]
                else:
                    # remove contradictory labels
                    if "forward" in labels and "backward" in labels:
                        labels.remove("forward")
                        labels.remove("backward")
                    if "left" in labels and "right" in labels:
                        labels.remove("left")
                        labels.remove("right")
                    # if there are no labels, assign label "none"
                    if len(labels) == 0:
                        labels = ["none"]
                label = ' '.join(labels)
                # rename corresponding image & copy to proper folder
                # example new file name: NE4 1.jpg
                if verbose:
                    print("moving {}.jpg to {}/{} {}.jpg".format(row_number, label, d, row_number))
                shutil.copy("../all data/{}/img/{}.jpg".format(d, row_number),
                    "./{}/{} {}.jpg".format(label, d, row_number))
                row_number += 1


if __name__ == "__main__":
    main()