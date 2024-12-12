-- Create databases
CREATE DATABASE lightrag;
CREATE DATABASE lightrag_test;
CREATE DATABASE lightrag_development;

-- Enable extensions in template1 so they are available in all new databases
\c template1

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- Set up each database
\c lightrag

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
SET search_path = ag_catalog, "$user", public;

\c lightrag_test

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
SET search_path = ag_catalog, "$user", public;

\c lightrag_development

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
SET search_path = ag_catalog, "$user", public;

-- Grant permissions (back in default database)
\c postgres

-- Grant permissions for all databases
GRANT ALL PRIVILEGES ON DATABASE lightrag TO postgres;
GRANT ALL PRIVILEGES ON DATABASE lightrag_test TO postgres;
GRANT ALL PRIVILEGES ON DATABASE lightrag_development TO postgres;

-- Set up AGE permissions for each database
\c lightrag
GRANT USAGE ON SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA ag_catalog TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON FUNCTIONS TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON SEQUENCES TO postgres;

\c lightrag_test
GRANT USAGE ON SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA ag_catalog TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON FUNCTIONS TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON SEQUENCES TO postgres;

\c lightrag_development
GRANT USAGE ON SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ag_catalog TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA ag_catalog TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON FUNCTIONS TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA ag_catalog GRANT ALL ON SEQUENCES TO postgres;

