# Contributing

## Adding features or fixing bugs

* Fork the repo
* Check out a feature or bug branch
* Add your changes
* Update README when needed
* Submit a pull request to upstream repo
* Add description of your changes
* Add and ensure tests are passing
* Ensure branch is mergeable
* Use Ruff to format and lint your code

## Testing

* Please make sure tests pass with `./tests`

## Doc building

* It is necessary to have valid credentials in `.earthdaily/credentials`
* inside docs/ : "sphinx-build -b html . html"

## Doc building on Lightning.ai

* Install the required sphinx dependencies in `requirements_dev.yml`
* Run `sphinx-build -b html docs/ docs/_build/`
* To view using the port extension run `python3 -m http.server --directory /teamspace/studios/this_studio/earthdaily-python-client/docs/_build 8000`
