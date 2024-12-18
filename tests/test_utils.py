"""Test utilities for LightRAG test suite.

This module provides utility functions used across different test modules in the LightRAG
test suite. It includes functions for method checking, configuration parsing, and
test cleanup operations.

Key Utilities:
- Method existence checking
- PostgreSQL URI parsing
- Configuration factory functions
- Standard cleanup procedures for test resources

The utilities are designed to be reusable across different storage implementations
and test scenarios, providing consistent behavior and resource management.
"""

import os
import shutil

def has_method(obj, method_name: str) -> bool:
    """Check if an object has a specific method implemented.

    This utility function verifies both the existence and callability of a method
    on a given object, useful for checking optional feature implementations.

    Args:
        obj: Object to check for method existence
        method_name: Name of the method to check

    Returns:
        bool: True if the method exists and is callable, False otherwise
    """
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))

def parse_postgres_uri(uri: str) -> dict:
    """Parse PostgreSQL URI into a configuration dictionary.

    Converts a standard PostgreSQL connection URI into a dictionary of connection
    parameters. Supports the format: postgresql://user:pass@host:5432/dbname

    Args:
        uri: PostgreSQL connection URI string

    Returns:
        dict: Configuration dictionary with keys:
            - user: Database user
            - password: User password
            - host: Database host
            - port: Database port (as integer)
            - database: Database name

    Raises:
        ValueError: If the URI format is invalid
    """
    import re
    pattern = r'postgresql://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)'
    match = re.match(pattern, uri)
    if not match:
        raise ValueError(f"Invalid postgres URI format: {uri}")
    config = match.groupdict()
    config['port'] = int(config['port'])
    return config

def postgres_config_factory():
    """Create a PostgreSQL configuration from environment variables.

    Uses POSTGRES_TEST_URI environment variable if available, otherwise falls back
    to default local development configuration.

    Returns:
        dict: PostgreSQL configuration dictionary
    """
    test_uri = os.getenv('POSTGRES_TEST_URI', 'postgresql://postgres:postgres@localhost:5432/lightrag_test')
    return parse_postgres_uri(test_uri)

async def standard_cleanup(store):
    """Perform standard cleanup operations for test storage instances.

    This function handles cleanup of various resources:
    - Removes working directories
    - Drops database tables/graphs
    - Closes database connections

    The function is storage-implementation aware and only performs
    operations that are supported by the given storage instance.

    Args:
        store: Storage instance to clean up
    """
    # Clean up working directory if it exists
    if hasattr(store, 'global_config') and 'working_dir' in store.global_config:
        working_dir = store.global_config['working_dir']
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

    # Drop tables/graphs if implemented
    if hasattr(store, 'drop'):
        await store.drop()

    # Close connections if implemented
    if hasattr(store, 'close'):
        await store.close()
