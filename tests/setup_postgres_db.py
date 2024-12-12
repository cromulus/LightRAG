import asyncio
import asyncpg
import os
import sys
from pathlib import Path

async def setup_postgres():
    """Set up PostgreSQL database and user for testing"""

    # Default superuser connection
    try:
        sys.stdout.write("Connecting to PostgreSQL as superuser... ")
        conn = await asyncpg.connect(
            user='postgres',
            password='postgres',
            database='postgres',
            host='localhost'
        )
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    try:
        # Create test database if it doesn't exist
        sys.stdout.write("Creating test database... ")
        await conn.execute('COMMIT')  # Exit any existing transaction
        await conn.execute('DROP DATABASE IF EXISTS lightrag_test')
        await conn.execute('CREATE DATABASE lightrag_test')
        print("Done!")

        # Create test user if it doesn't exist
        sys.stdout.write("Creating test user... ")
        await conn.execute('''
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'lightrag_test') THEN
                    CREATE USER lightrag_test WITH PASSWORD 'lightrag_test';
                END IF;
            END
            $$;
        ''')
        print("Done!")

        # Grant privileges
        sys.stdout.write("Granting privileges... ")
        await conn.execute('''
            GRANT ALL PRIVILEGES ON DATABASE lightrag_test TO lightrag_test;
            ALTER DATABASE lightrag_test OWNER TO lightrag_test;
        ''')
        print("Done!")

        await conn.close()

        # Connect to test database to create extensions and set permissions
        sys.stdout.write("Connecting to test database... ")
        test_conn = await asyncpg.connect(
            user='postgres',
            password='postgres',
            database='lightrag_test',
            host='localhost'
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
        await test_conn.execute('''
            GRANT ALL ON SCHEMA public TO lightrag_test;
            ALTER SCHEMA public OWNER TO lightrag_test;
            GRANT USAGE ON SCHEMA ag_catalog TO lightrag_test;
            GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ag_catalog TO lightrag_test;
            GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ag_catalog TO lightrag_test;
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
