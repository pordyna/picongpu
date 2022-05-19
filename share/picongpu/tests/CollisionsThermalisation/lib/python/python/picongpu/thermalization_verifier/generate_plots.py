import argparse
import os
from ThermalizationVerifier import ThermalizationVerifier


def main():
    DEFAULT_ABS_TOLERANCE = 1e-6
    DEFAULT_REL_TOLERANCE = 1e-6
    DEFAULT_ABS_THRESHOLD = 1e-6
    parser = argparse.ArgumentParser(description="It calculates electron and "
                                                 "ion temperatures for all simulation steps and plots them together with a theretical curve.")
    parser.add_argument('dir', nargs='?', help="simulation directory containing the simOutput directory", default=os.getcwd())
    parser.add_argument("--coulomb_log", help="Coulomb logarithm for theoretical calculation. If not set it uses "
                                              "dynamic caluclation based on a theretical model for electron ion collisions", type=float)
    parser.add_argument("--file", help="figure file name", type=str)
    args = parser.parse_args()

    verifier = ThermalizationVerifier(args.dir)
    verifier.calculate_temperatures()
    verifier.calculate_theretical_values(args.coulomb_log)
    verifier.plot(to_file=True, file_name=args.file)


if __name__ == '__main__':
    main()
