FROM mcr.microsoft.com/devcontainers/python:3.10

# Add PostgreSQL's repository
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/postgresql.list

# Install PostgreSQL client tools and build dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        postgresql-client \
        build-essential \
        git \
        cmake \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        flex \
        bison \
        dnsutils \
        iputils-ping \
        libpq-dev \
        rustc \
        cargo \
        direnv \
        ca-certificates \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Create the workspace directory
WORKDIR /workspace

# Copy both requirements files
COPY requirements.txt /tmp/requirements.txt
COPY .devcontainer/requirements.txt /tmp/dev-requirements.txt

# Install requirements
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/dev-requirements.txt

# Set up direnv
RUN echo 'eval "$(direnv hook bash)"' >> /etc/bash.bashrc
