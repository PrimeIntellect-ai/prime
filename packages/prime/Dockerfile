# Prime CLI container image
# This image packages the released Prime CLI together with uv so it can be used in
# automation scenarios.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG PRIME_VERSION
ENV PRIME_VERSION=${PRIME_VERSION}
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        bash \
        git \
        build-essential \
        gcc \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY dist/ /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl \
    && rm -rf /tmp/dist

RUN prime --version

WORKDIR /workspace

CMD ["prime", "--help"]
