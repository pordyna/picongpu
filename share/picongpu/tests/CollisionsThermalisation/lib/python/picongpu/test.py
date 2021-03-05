import argparse
from os import path
from ThermalizationVerifier import ThermalizationVerfier


def main():
    DEFAULT_ABS_TOLERANCE = 1e-6
    DEFAULT_REL_TOLERANCE = 1e-6
    DEFAULT_ABS_THRESHOLD = 1e-6
    parser = argparse.ArgumentParser(description="Verifies a CollisionThermalization test. It calculates electron and "
                                                 "ion temperatures for all simulation steps and compares with stored "
                                                 "reference data. Two values are considered close if either their "
                                                 "relative error is below the rel_tolerance value or their absolute"
                                                 " difference is below the abs_tolerance and at least one value is "
                                                 "smaller than the abs_threshold.")
    parser.add_argument("--abs_tolerance", help="Absolut tolerance used to compare values", type=float,
                        default=DEFAULT_ABS_TOLERANCE)
    parser.add_argument("--rel_tolerance", help="Relative tolerance used to compare values", type=float,
                        default=DEFAULT_REL_TOLERANCE)
    parser.add_argument("--abs_threshold", help="Value threshold for the absolute error", type=float,
                        default=DEFAULT_ABS_THRESHOLD)
    args = parser.parse_args()

    verifier = ThermalizationVerfier(path.abspath("../../../"))
    verifier.calculate_temperatures()
    verifier.load_reference()
    electrons, ions = verifier.compare(args.abs_tolerance, args.abs_threshold, args.rel_tolerance)
    if electrons and ions:
        print("PASSED")
    else:
        print("FAILED")
        if not electrons:
            print("Electron temperatures don't match the reference!")
        if not ions:
            print("Ion temperatures don't match the reference!")


if __name__ == '__main__':
    main()
