import sys
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(
                    prog="predict.py",
                    description="predict.py is a script to predict the genre for a given audio files")
    
    parser.add_argument("--input", required=True, action="store", help="path to the folder with audio files")
    parser.add_argument("--model", required=True, action="store", help="path to the neural network model")
    parser.add_argument("--type", required=False, action="store", choices=["min", "max", "mean"], default="mean", help="type of aggregation for the segments")
    parser.add_argument("--count", required=False, action="store", type=int, choices=range(1,11), default=-1, help="number of most likely genres to display")

    return parser.parse_args(args)
