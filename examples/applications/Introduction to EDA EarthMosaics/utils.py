import numpy as np
import requests


def get_new_token(session, token_url):
    """
    Obtain a new authentication token using client credentials
    """
    try:
        token_response = session.post(token_url, data={"grant_type": "client_credentials"})
        token_response.raise_for_status()
        tokens = token_response.json()
        return tokens["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Failed to obtain token: {e}")
        return


def replace_alternate_href(item):
    """
    Replaces S3 HREFs in an item with its alternate public HTTPS HREFs
    """
    for asset in item.assets:
        alt_href = item.assets[asset].extra_fields["alternate"]["download"]["href"]
        item.assets[asset].href = alt_href
    return item


def linear_stretch_rgb(rgb_array, pct_lower=2, pct_upper=98):
    """
    Apply a linear percent stretch on RGB data, default 2%-98%
    INPUTS:
    * RGB Array of shape (ny, nx, nbands)
    * pct_lower bounds
    * pct_upper bounds

    OUTPUTS:
    * Stretched RGB Array of shape (ny, nx, nbands) of type UINT8
    """
    stretched = rgb_array.copy().astype(float)
    for i in range(stretched.shape[-1]):
        lower, upper = np.percentile(stretched[:, :, i], [pct_lower, pct_upper])
        band = np.clip((stretched[:, :, i] - lower) * (255.0 / (upper - lower)), 0, 255)
        stretched[:, :, i] = band

    return stretched.astype(np.uint8)
