import os


def login_huggingface_from_env(env_var: str = "HUGGINGFACE_TOKEN", *, verbose: bool = True) -> bool:
    """
    Log in to Hugging Face using a token from an environment variable.

    Returns:
        True if login was attempted and succeeded, otherwise False.
    """
    token = os.getenv(env_var)
    if not token:
        return False

    try:
        from huggingface_hub import login
    except Exception as e:
        if verbose:
            print(f"Could not import huggingface_hub.login: {e}")
        return False

    try:
        login(token)
        if verbose:
            print("Success logging in to Hugging Face!")
        return True
    except Exception as e:
        if verbose:
            print(f"Login error: {e}")
        return False


def list_image_files(indir, sort=False):
    """
    List all image files in a directory and its subdirectories.

    Args:
        indir: Directory path to search for image files
        sort: If True, sort the resulting file paths alphabetically

    Returns:
        List of image file paths with extensions: jpg, jpeg, png, bmp
    """
    image_files = []
    goodexts = ['jpg', 'jpeg', 'png', 'bmp']

    for root, dirs, files in os.walk(indir):
        for file in files:
            if file.split('.')[-1].lower() in goodexts:
                image_files.append(os.path.join(root, file))

    if sort:
        image_files.sort()

    return image_files