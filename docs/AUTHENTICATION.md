### Authentication

Advanced users can use the `earthdaily/utils/copy_credentials_template.py` script to generate credentials file as JSON, TOML or .env. Alternatively use one of the approaches listed below.

#### Authentication from the default credentials file

Using a TOML file for authentication has the advantage that the authentication is handled seamlessly by the client and it is not necessary to point to the authentication file. 
A TOML credentials file will be created in the following locations:

* "$HOME/.earthdaily/credentials" on linux
* "$USERPROFILE/.earthdaily/credentials" on Windows

Run from the root of the repository:
```console
copy-earthdaily-credentials-template --default
```

Edit it to insert your credentials.
The following code will automatically find and use the credentials for authentification.

```python
from earthdaily import EarthDataStore
eds = EarthDataStore()
```

#### Authentication from a JSON file

Authentication credentials can be given as an input JSON file.
You can generate a JSON credentials file with the following command:

```console
copy-earthdaily-credentials-template --json "/path/to/the/credentials_file.json"
```

Edit it to insert your credentials.
Then use it as input for authentification:

```python
from pathlib import Path
from earthdaily import EarthDataStore
eds = EarthDataStore(json_path = Path("/path/to/the/credentials_file.json"))
```

#### Authentication from a TOML file

Authentication credentials can be given as input with a TOML file.
You can generate a TOML credentials file with the following command:

```console
copy-earthdaily-credentials-template --toml "/path/to/the/credentials_file"
```

Edit it to insert your credentials.
Then use it as input for authentification:

```python
from pathlib import Path
from earthdaily import EarthDataStore
eds = EarthDataStore(toml_path = Path("/path/to/the/credentials_file"))
```

#### Authentication from Environment Variables

Authentication credentials can be automatically parsed from environment variables.
The [python-dotenv](https://github.com/theskumar/python-dotenv) package is supported for convenience.

Rename the `.env.sample` file in this repository to `.env` and enter your Earth Data Store authentication credentials. 
Note this file is gitingored and will not be committed.

In your script or notebook, add:

```python
from dotenv import load_dotenv

load_dotenv(".env")  # Load environment variables from .env file
```