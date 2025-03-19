"""Configuration module for earthdaily package.

This module provides a centralized configuration system for the earthdaily package,
allowing users to set and retrieve options that affect package behavior.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OptionDef:
    """Definition of a configuration option.

    Parameters
    ----------
    default : Any
        The default value for the option.
    description : str
        A description of what the option controls.
    validator : Optional[Callable[[Any], bool]], optional
        A function that validates values for this option, by default None.
    valid_values : Optional[List[Any]], optional
        A list of valid values for this option, by default None.
    """

    default: Any
    description: str
    validator: Optional[Callable[[Any], bool]] = None
    valid_values: Optional[List[Any]] = None

    def validate(self, value: Any) -> bool:
        """Validate a value against this option's constraints.

        Parameters
        ----------
        value : Any
            The value to validate.

        Returns
        -------
        bool
            True if validation passes.

        Raises
        ------
        ValueError
            If the value fails validation.
        """
        if self.validator is not None:
            if not self.validator(value):
                raise ValueError(
                    f"Invalid value: {value}. "
                    f"Must satisfy validator: {self.validator.__doc__ or 'no description'}"
                )
        if self.valid_values is not None:
            if value not in self.valid_values:
                raise ValueError(
                    f"Invalid value: {value}. Must be one of: {self.valid_values}"
                )
        return True


# Define all available options
OPTIONS: Dict[str, OptionDef] = {
    "groupby_date_engine": OptionDef(
        default="numpy",
        description="Engine to use for grouping by date",
        valid_values=["numpy", "numba", "numbagg", "flox"],
    ),
    "disable_known_warning": OptionDef(
        default=True,
        description="Disable known warning from dependencies (value in cast)",
        valid_values=[True, False],
    ),
}


class Options:
    """Options manager for earthdaily package.

    This class manages global configuration options that can be accessed
    through both attribute-style access (options.option_name) and
    function-style access (get_option('option_name')).

    Methods
    -------
    get_option(key)
        Get the value of an option.
    set_option(key, value)
        Set the value of an option.
    reset_option(key)
        Reset an option to its default value.
    describe_option(key)
        Get a description of an option.
    list_options()
        List all available options with their descriptions.
    """

    def __init__(self):
        # Store the actual values
        self._options = {key: opt.default for key, opt in OPTIONS.items()}

        # Create properties for each option
        for key in OPTIONS:
            setattr(
                Options,
                key,
                property(
                    fget=lambda self, _key=key: self._options[_key],
                    fset=lambda self, value, _key=key: self._set_option(_key, value),
                    doc=OPTIONS[key].description,
                ),
            )

    def _set_option(self, key: str, value: Any) -> None:
        """Internal method to set an option value with validation.

        Parameters
        ----------
        key : str
            The name of the option to set.
        value : Any
            The value to set.

        Raises
        ------
        KeyError
            If the option doesn't exist.
        ValueError
            If the value is invalid.
        """
        if key not in OPTIONS:
            raise KeyError(f"Unknown option: {key}")

        option_def = OPTIONS[key]
        option_def.validate(value)
        self._options[key] = value

    def get_option(self, key: str) -> Any:
        """Get the value of an option.

        Parameters
        ----------
        key : str
            The name of the option to retrieve.

        Returns
        -------
        Any
            The value of the option.

        Raises
        ------
        KeyError
            If the option doesn't exist.
        """
        if key not in self._options:
            raise KeyError(f"Unknown option: {key}")
        return self._options[key]

    def set_option(self, key: str, value: Any) -> None:
        """Set the value of an option.

        Parameters
        ----------
        key : str
            The name of the option to set.
        value : Any
            The value to set the option to.

        Raises
        ------
        KeyError
            If the option doesn't exist.
        ValueError
            If the value is invalid.
        """
        self._set_option(key, value)

    def reset_option(self, key: str) -> None:
        """Reset an option to its default value.

        Parameters
        ----------
        key : str
            The name of the option to reset.

        Raises
        ------
        KeyError
            If the option doesn't exist.
        """
        if key not in OPTIONS:
            raise KeyError(f"Unknown option: {key}")

        self._options[key] = OPTIONS[key].default

    def describe_option(self, key: str) -> str:
        """Get a description of an option.

        Parameters
        ----------
        key : str
            The name of the option to describe.

        Returns
        -------
        str
            A string describing the option, including its current value,
            default value, and constraints.

        Raises
        ------
        KeyError
            If the option doesn't exist.
        """
        if key not in OPTIONS:
            raise KeyError(f"Unknown option: {key}")

        opt_def = OPTIONS[key]
        current_value = self._options[key]

        description = [
            f"Option: {key}",
            f"Description: {opt_def.description}",
            f"Current value: {current_value}",
            f"Default value: {opt_def.default}",
        ]

        if opt_def.valid_values is not None:
            description.append(f"Valid values: {opt_def.valid_values}")
        if opt_def.validator is not None:
            description.append(
                f"Validator: {opt_def.validator.__doc__ or 'no description'}"
            )

        return "\n".join(description)

    def list_options(self) -> str:
        """List all available options with their descriptions.

        Returns
        -------
        str
            A formatted string containing descriptions of all options.
        """
        return "\n".join(key for key in sorted(OPTIONS))


# Global instance
options = Options()


# Public API functions
def get_option(key: str) -> Any:
    """Get the value of an option.

    Parameters
    ----------
    key : str
        The name of the option to retrieve.

    Returns
    -------
    Any
        The value of the option.

    Raises
    ------
    KeyError
        If the option doesn't exist.
    """
    return options.get_option(key)


def set_option(key: str, value: Any) -> None:
    """Set the value of an option.

    Parameters
    ----------
    key : str
        The name of the option to set.
    value : Any
        The value to set the option to.

    Raises
    ------
    KeyError
        If the option doesn't exist.
    ValueError
        If the value is invalid.
    """
    options.set_option(key, value)


def reset_option(key: str) -> None:
    """Reset an option to its default value.

    Parameters
    ----------
    key : str
        The name of the option to reset.

    Raises
    ------
    KeyError
        If the option doesn't exist.
    """
    options.reset_option(key)


def describe_option(key: str) -> str:
    """Get a description of an option.

    Parameters
    ----------
    key : str
        The name of the option to describe.

    Returns
    -------
    str
        A string describing the option, including its current value,
        default value, and constraints.

    Raises
    ------
    KeyError
        If the option doesn't exist.
    """
    return options.describe_option(key)


def list_options() -> str:
    """List all available options with their descriptions.

    Returns
    -------
    str
        A formatted string containing descriptions of all options.
    """
    return options.list_options()
