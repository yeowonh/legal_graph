<<<<<<< HEAD

""""

# To run this script from the command line, use:
# python <script_name>.py --path <path_to_json_files> --save_path <path_to_save_merged_json> --filtering <file_extension_to_filter_by>
# Example: python merge_json.py --path "../../results" --save_path "../../results/merged_DCM_data.json" --filtering "_clause.json"
"""


import sys, os
import argparse
import utils

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Argument group
parser = argparse.ArgumentParser(description="Merge JSON files.")
g = parser.add_argument_group("Arguments")

g.add_argument("--path", type=str, default="../../results", help="Path to the directory containing JSON files.")
g.add_argument("--save_path", type=str, default="../../results/merged_DCM_1-2_data.json", help="Path where the merged JSON will be saved.")
g.add_argument("--filtering", type=str,  default="_clause.json", help="File extension to filter by.")

args = parser.parse_args()


def main(args):
    # Run the merge function with arguments
    data = utils.merge_to_json(args.path, args.save_path, args.filtering)
    print("## data[0] 예시\n",data[0])

if __name__ == "__main__":
    exit(main(parser.parse_args()))
=======

""""

# To run this script from the command line, use:
# python <script_name>.py --path <path_to_json_files> --save_path <path_to_save_merged_json> --filtering <file_extension_to_filter_by>
# Example: python merge_json.py --path "../../results" --save_path "../../results/merged_DCM_data.json" --filtering "_clause.json"
"""


import sys, os
import argparse
import utils

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Argument group
parser = argparse.ArgumentParser(description="Merge JSON files.")
g = parser.add_argument_group("Arguments")

g.add_argument("--path", type=str, default="../../results", help="Path to the directory containing JSON files.")
g.add_argument("--save_path", type=str, default="../../results/merged_DCM_1-2_data.json", help="Path where the merged JSON will be saved.")
g.add_argument("--filtering", type=str,  default="_clause.json", help="File extension to filter by.")

args = parser.parse_args()


def main(args):
    # Run the merge function with arguments
    data = utils.merge_to_json(args.path, args.save_path, args.filtering)
    print("## data[0] 예시\n",data[0])

if __name__ == "__main__":
    exit(main(parser.parse_args()))
>>>>>>> 769c3460961afd05bd9ea986d27e87b1abdb89e7
