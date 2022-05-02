from os import get_inheritable
import yaml
import argparse

default_config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument(
    '-c',
    '--config_yaml',
    default=
    './config/test.yaml',
    type=str,
    metavar='FILE',
    help='YAML config file specifying default arguments')


# YAML should override the argparser's content
def _parse_args_and_yaml(given_parser=None):
    if given_parser == None:
        given_parser = default_config_parser
    given_configs, remaining = given_parser.parse_known_args()
    if given_configs.config_yaml:
        with open(given_configs.config_yaml, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            given_parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = given_parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def parse_args_and_yaml(arg_parser=None):
    return _parse_args_and_yaml(arg_parser)[0]


if __name__ == "__main__":
    args, args_text = _parse_args_and_yaml()
    print('1')