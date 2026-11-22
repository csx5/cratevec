FROM rust:1.83-slim AS builder
WORKDIR /app
COPY Cargo.toml ./
COPY crates ./crates
RUN cargo build --release -p cratevec-server -p cratevec-cli

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/cratevec-server /usr/local/bin/cratevec-server
COPY --from=builder /app/target/release/cratevec /usr/local/bin/cratevec
COPY ui ./ui
VOLUME ["/app/data"]
EXPOSE 7420
ENV CRATEVEC_PORT=7420
CMD ["cratevec-server"]
