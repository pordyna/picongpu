import argparse
import os
from ThermalizationVerifier import ThermalizationVerifier

def main():
    parser = argparse.ArgumentParser(description="Saves reference data for the CollisionThermalization test. "
                                                 "It calculates electron and ion temperatures for all simulation steps"
                                                  "and saves in the local directory.")
    parser.add_argument('dir', nargs='?', help="simulation directory containing the simOutput directory", default=os.getcwd())
    args = parser.parse_args()
    verifier = ThermalizationVerifier(args.dir)
    verifier.calculate_temperatures()
    verifier.save_reference()


if __name__ == '__main__':
    main()
