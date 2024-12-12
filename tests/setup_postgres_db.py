import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

def parse_postgres_uri(uri):
    """Parse PostgreSQL URI into connection parameters"""
    parsed = urlparse(uri)
    return {
        'user': parsed.username,
        'password': parsed.password,
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path[1:] if parsed.path else 'postgres'
    }

async def setup_postgres():
    """Set up PostgreSQL database and user for testing"""
    # Get connection parameters from URIs
    admin_uri = os.getenv('POSTGRES_URI', 'postgresql://postgres:postgres@db:5432/postgres')
    test_uri = os.getenv('POSTGRES_TEST_URI', 'postgresql://lightrag_test:lightrag_test@db:5432/lightrag_test')

    admin_config = parse_postgres_uri(admin_uri)
    test_config = parse_postgres_uri(test_uri)

    # Default superuser connection
    try:
        sys.stdout.write("Connecting to PostgreSQL as superuser... ")
        conn = await asyncpg.connect(**admin_config)
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    try:
        # Create test database if it doesn't exist
        sys.stdout.write("Creating test database... ")
        await conn.execute('COMMIT')  # Exit any existing transaction
        await conn.execute(f'DROP DATABASE IF EXISTS {test_config["database"]}')
        await conn.execute(f'CREATE DATABASE {test_config["database"]}')
        print("Done!")

        # Create test user if it doesn't exist
        sys.stdout.write("Creating test user... ")
        await conn.execute(f'''
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = $1) THEN
                    CREATE USER {test_config["user"]} WITH PASSWORD $2;
                END IF;
            END
            $$;
        ''', test_config["user"], test_config["password"])
        print("Done!")

        # Grant privileges
        sys.stdout.write("Granting privileges... ")
        await conn.execute(f'''
            GRANT ALL PRIVILEGES ON DATABASE {test_config["database"]} TO {test_config["user"]};
            ALTER DATABASE {test_config["database"]} OWNER TO {test_config["user"]};
        ''')
        print("Done!")

        await conn.close()

        # Connect to test database to create extensions and set permissions
        sys.stdout.write("Connecting to test database... ")
        test_conn = await asyncpg.connect(
            **{**admin_config, 'database': test_config['database']}
        )
        print("Connected!")

        # Create required extensions
        sys.stdout.write("Creating extensions... ")
        await test_conn.execute('''
            -- Enable required extensions
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS age;

            -- Initialize AGE extension
            LOAD 'age';
            SET search_path = ag_catalog, "$user", public;
        ''')
        print("Done!")

        # Grant additional permissions needed for AGE and public schema
        sys.stdout.write("Granting permissions... ")
        await test_conn.execute(f'''
            GRANT ALL ON SCHEMA public TO {test_config["user"]};
            ALTER SCHEMA public OWNER TO {test_config["user"]};
            GRANT USAGE ON SCHEMA ag_catalog TO {test_config["user"]};
            GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ag_catalog TO {test_config["user"]};
            GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ag_catalog TO {test_config["user"]};
            GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA ag_catalog TO {test_config["user"]};
            GRANT ALL PRIVILEGES ON ALL ROUTINES IN SCHEMA ag_catalog TO {test_config["user"]};
            ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON TABLES TO {test_config["user"]};
            ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON FUNCTIONS TO {test_config["user"]};
            ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON SEQUENCES TO {test_config["user"]};
            ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON TYPES TO {test_config["user"]};
        ''')
        print("Done!")

        await test_conn.close()
        print("\nDatabase setup completed successfully!")

    except Exception as e:
        print(f"Error during setup: {e}")
    finally:
        if 'conn' in locals():
            await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_postgres())
