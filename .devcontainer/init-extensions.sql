-- Enable extensions in template1 so they are available in all new databases
\c template1

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- Initialize AGE
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Switch to default database
\c postgres

-- Create extensions in default database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- Initialize AGE
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
