import os

def has_method(obj, method_name: str) -> bool:
    """Check if an object has a specific method implemented"""
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))


def parse_postgres_uri(uri: str) -> dict:
    """Parse postgres URI into config dict"""
    # Example: postgresql://user:pass@host:5432/dbname
    import re
    pattern = r'postgresql://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)'
    match = re.match(pattern, uri)
    if not match:
        raise ValueError(f"Invalid postgres URI format: {uri}")
    config = match.groupdict()
    config['port'] = int(config['port'])
    return config  # Return config dict directly

def postgres_config_factory():
    test_uri = os.getenv('POSTGRES_TEST_URI', 'postgresql://postgres:postgres@localhost:5432/lightrag_test')
    return parse_postgres_uri(test_uri)
