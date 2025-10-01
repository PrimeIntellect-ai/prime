# Prime CLI container image
# This image packages the released Prime CLI together with uv so it can be used in
# automation scenarios.

FROM python:3.11-slim AS runtime

ARG PRIME_VERSION
ENV PRIME_VERSION=${PRIME_VERSION}
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Install runtime dependencies and uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl bash \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy in the built wheel from the release workflow and install it
COPY dist/ /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl \
    && rm -rf /tmp/dist

# Basic sanity check so the build fails if the CLI is not installed correctly
RUN prime --version

WORKDIR /workspace

CMD ["prime", "--help"]
