# FFT comparison

Measured median milliseconds. FFT packing is not the 9/4 algorithm.

| n | NumPy | FFT packing | Strassen leaf 32 | FFT points |
|---|---:|---:|---:|---:|
| 32 | 0.0072 | 1.4901 | 0.0083 | 32768 |
| 64 | 0.0191 | 20.9855 | 0.1039 | 262144 |
| 128 | 0.1189 | 199.2055 | 0.7169 | 2097152 |

## Proposition 3.1 Fourier selector

| M | Direct root sum | FFT selector |
|---|---:|---:|
| 4 | 0.5469 | 0.0768 |
| 8 | 10.9049 | 0.1411 |
| 16 | 342.4356 | 2.7953 |

66 functional checks passed; all timed outputs also checked.
Projection acceleration does not synthesize a fast matrix rank decomposition.
Strassen at n=32 with leaf=32 is a leaf-only control, not recursive Strassen.
