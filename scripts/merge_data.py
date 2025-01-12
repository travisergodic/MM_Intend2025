import json
import argparse


def main():
    data = []
    for json_file in args.json_list:
        with open(json_file, "r") as f:
            data += json.load(f)

    with open(args.save_path, "w") as f:
        json.dump(data, f)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json_list", type=str, nargs="+")
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main()