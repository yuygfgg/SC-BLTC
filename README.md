# SC-BLTC (Stream Cipher Blind Time Long Code)

A Rust implementation of the SC-BLTC spread-spectrum communication protocol, designed for covert, low-probability-of-intercept (LPI) HF radio communications.

### Security Properties

1. **Time-Blindness**: The transmission timestamp is embedded implicitly in the spreading code seed, requiring the receiver to perform a blind time search
2. **Cryptographic Spreading**: Each transmission uses a unique AES-CTR keystream based on the 256-bit shared key and millisecond-resolution timestamp
3. **Low Power Spectral Density**: By spreading the signal over a 5 kHz bandwidth, its power spectral density at operational SNR can be below the noise floor, which hinders detection by simple energy-based receivers.

> [!WARNING]
> The fixed chip rate (5 kcps) presents a vulnerability that sophisticated adversaries could exploit using cyclostationary feature detection. Therefore, while the system offers a degree of covertness against non-targeted surveillance, it is not fully undetectable.

## Building

```bash
cargo build --release
```

## Usage (TCP I/Q Mode)

This implementation uses TCP for I/Q sample streaming.

### Start Receiver

```bash
cargo run --release --bin rx -- [parameters]
```

### Start Transmitter

```bash
cargo run --release --bin tx -- [parameters]
```

### Transmitter Parameters (`tx`)

| Argument                 | Default          | Description                                                |
| :----------------------- | :--------------- | :--------------------------------------------------------- |
| `--addr`                 | `127.0.0.1:5555` | Receiver address                                           |
| `--key-hex`              | `00..00`         | 32-byte key in hex                                         |
| `--ver`                  | `1`              | Protocol version (Header.Ver)                              |
| `--typ`                  | `1`              | Message type (Header.Type)                                 |
| `--frames`               | `0`              | Send N typed frames then exit (0 = infinite)               |
| `--lead-s`               | `0.5`            | Initial silence before the first frame (seconds)           |
| `--gap-s`                | `0.1`            | Extra silence between frames (seconds)                     |
| `--cfo-hz`               | `0.0`            | Simulated carrier frequency offset (Hz)                    |
| `--sro-ppm`              | `0.0`            | Sampling-rate offset (ppm). Positive values = fast clock   |
| `--doppler-std-hz`       | `0.0`            | Random Doppler stddev (Hz)                                 |
| `--doppler-tau-s`        | `0.5`            | Doppler correlation time constant (seconds)                |
| `--doppler-max-hz`       | `0.0`            | Clamp absolute Doppler to this value (Hz)                  |
| `--mp-tap`               | `[]`             | Explicit multipath taps: `delay_samp,gain_db[,phase_deg]`  |
| `--mp-paths`             | `0`              | Enable random multipath with N paths (0 = off)             |
| `--mp-max-delay-samp`    | `40`             | Max extra delay for random multipath taps (samples)        |
| `--mp-decay-db-per-samp` | `0.20`           | Power delay profile slope for random multipath (dB/sample) |
| `--noise-std`            | `0.0`            | AWGN stddev per I/Q component (linear)                     |
| `--amp`                  | `1.0`            | Output amplitude scaling                                   |
| `--seed`                 | `1`              | RNG seed                                                   |
| `--start-delay-s`        | `0.2`            | Stream start time offset from "now" (seconds)              |
| `--write-timeout-ms`     | `0`              | TCP write timeout in milliseconds (0 = no timeout)         |
| `--tcp-delay-ms`         | `0`              | Simulated TCP egress delay (milliseconds)                  |

### Receiver Parameters (`rx`)

| Argument            | Default          | Description                                       |
| :------------------ | :--------------- | :------------------------------------------------ |
| `--bind`            | `127.0.0.1:5555` | Bind address                                      |
| `--key-hex`         | `00..00`         | 32-byte key in hex                                |
| `--ldpc-maxiter`    | `500`            | LDPC max iterations                               |
| `--w-sec`           | `0.5`            | Blind acquisition time window W (seconds)         |
| `--p-fa-total`      | `1e-9`           | CFAR target false-alarm probability               |
| `--n-finger`        | `3`              | Number of RAKE fingers to initialize              |
| `--buffer-sec`      | `30.0`           | Ring buffer length in seconds                     |
| `--acq-interval-ms` | `250`            | How often to attempt acquisition (milliseconds)   |
| `--read-timeout-ms` | `0`              | TCP read timeout in milliseconds (0 = no timeout) |
| `--debug`           | `false`          | Enable verbose debug output                       |

## References

- See [Specification.pdf](Specification.pdf) for the complete protocol specification
