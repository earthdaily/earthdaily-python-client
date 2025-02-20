# Earth Data Store Client

The Earth Data Store (EDS) Client is a Python library for interacting with the EarthDaily Analytics Earth Data Store API. It provides a simple interface for authentication, making API requests, and handling responses.

## Installation

You can install the EDS Client using pip:

```bash
pip install earthdaily
```

## Environment Variables

Before running the script, ensure the following environment variables are set:

- **`EDS_CLIENT_ID`**: Your application's Client ID for authenticating with the Earth Data Store.
- **`EDS_SECRET`**: The Client Secret associated with your Client ID.
- **`EDS_AUTH_URL`**: The URL used for obtaining authentication tokens.
- **`EDS_API_URL`**: The URL used for interacting with the API’s endpoints.

Alternatively, if you prefer not to use environment variables, you can directly pass these values through the `EDSConfig` in your script:

```python
client = EDSClient(EDSConfig(client_id="your_client_id_here", client_secret="your_client_secret_here", token_url="https://your-token-url.com"))
```

## Usage

TBD

## Documentation

For more detailed documentation, please refer to the [official documentation](https://earthdaily.github.io/EDA-Documentation/).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any problems or have any questions, please [open an issue](https://github.com/earthdaily/earthdaily-python-client/issues/new) on our GitHub repository.
