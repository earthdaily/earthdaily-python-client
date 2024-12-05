"""
Utility script used for copying credentials templates to a user-defined path
"""

import click
import dotenv
import json
import toml
from pathlib import Path

default_path = Path.home() / ".earthdaily/credentials"

default_configuration = {
    "EDS_AUTH_URL": "https://..",
    "EDS_CLIENT_ID": "123",
    "EDS_SECRET": "123",
}


def write_json(json_path: Path) -> None:
    """
    Write template JSON credentials file.

    Parameters
    ----------
    json_path : Path
        Path to output JSON file.
    """
    print(f"Try to write credentials template to {json_path}")

    if json_path.exists():
        print(f"{json_path} already exists")
        return

    with json_path.open("w") as f:
        json.dump(default_configuration, f)

    print(f"Credentials file written to {json_path}")
    print("Please edit it to insert your credentials")


def write_toml(toml_path: Path) -> None:
    """
    Write template TOML credentials file.

    Parameters
    ----------
    toml_path : Path
        Path to output TOML file.
    """
    print(f"Try to write credentials template to {toml_path}")

    if toml_path.exists():
        print(f"{toml_path} already exists")
        return

    with toml_path.open("w") as f:
        toml.dump({"default": default_configuration}, f)

    print(f"Credentials file written to {toml_path}")
    print("Please edit it to insert your credentials")


def write_env(env_path: Path) -> None:
    """
    Write template .env credentials file.

    Parameters
    ----------
    env_path : Path
        Path to output .env file.
    """
    print(f"Try to write credentials template to {env_path}")

    if env_path.exists():
        print(f"{env_path} already exists")
        return

    with env_path.open("w") as f:
        for key, value in default_configuration.items():
            line = f'{key}="{value}"\n'
            f.write(line)

    print(f"Credentials file written to {env_path}")
    print("Please edit it to insert your credentials")


@click.command("Copy credentials templates in all accepted formats")
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    help="Path to the output JSON file containing the credentials keys (but no values)",
)
@click.option(
    "--toml",
    "toml_path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    help="Path to the output TOML file containing the credentials keys (but no values)",
)
@click.option(
    "--env",
    "env_path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    help="Path to the output .env file containing the credentials keys (but no values)",
)
@click.option(
    "--default",
    "default",
    is_flag=True,
    show_default=True,
    default=False,
    help=f"Copy the TOML template to {default_path}, with credential keys (and no values)",
)
def cli(json_path: Path, toml_path: Path, env_path: Path, default: bool) -> None:
    if json_path is not None:
        write_json(json_path=json_path)

    if toml_path is not None:
        write_toml(toml_path=toml_path)

    if env_path is not None:
        write_env(env_path=env_path)

    if default:
        default_path.parent.mkdir(exist_ok=True, parents=True)
        write_toml(toml_path=default_path)


if __name__ == "__main__":
    cli()
