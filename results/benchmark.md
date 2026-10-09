# Measured matrix multiplication benchmark

Median milliseconds; blank = omitted. Paper 9/4 algorithm has no runnable certificate.

| n | numpy | naive | outer | blocked_128 | strassen_32 | strassen_64 | strassen_128 | strassen_256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.0074 | 6.1157 | 0.1596 | 0.0129 |  |  |  |  |
| 64 | 0.0206 | 47.7029 | 0.5342 | 0.0288 | 0.0974 |  |  |  |
| 128 | 0.1037 | 367.3001 | 3.1259 | 0.1253 | 0.7730 | 0.2387 |  |  |
| 256 | 0.6966 |  | 19.7139 | 1.0195 | 7.1388 | 2.7686 | 1.7950 |  |
| 512 | 6.3465 |  | 433.8910 | 9.9175 | 50.3096 | 26.3170 | 17.3143 | 10.3521 |
| 1024 | 44.1958 |  | 4586.9963 | 75.6599 | 424.4247 | 206.1678 | 143.9940 | 97.8113 |

Verification: 286 checks passed.

Strassen leaves use BLAS; only leaf sizes smaller than n are timed.
See JSON for raw samples, errors, environment and backend details.
