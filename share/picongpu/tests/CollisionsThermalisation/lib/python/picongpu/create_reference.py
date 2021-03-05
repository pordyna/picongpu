import argparse
from os import path
from ThermalizationVerifier import ThermalizationVerfier


def main():
    parser = argparse.ArgumentParser(description="Saves reference data for the CollisionThermalization test. "
                                                 "It calculates electron and ion temperatures for all simulation steps"
                                                 "and saves in the local directory.")

    verifier = ThermalizationVerfier(path.abspath("../../../"))
    verifier.calculate_temperatures()
    verifier.save_reference()


if __name__ == '__main__':
    main()
