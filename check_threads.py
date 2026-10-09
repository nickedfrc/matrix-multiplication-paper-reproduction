# SPDX-License-Identifier: Apache-2.0
# Developed with OpenAI Codex assistance; see NOTICE for mathematical sources.
import ctypes, json
from pathlib import Path
import numpy as np
info = []
for path in (Path(np.__file__).parent.parent/"numpy.libs").glob("*openblas*.dll"):
    lib = ctypes.CDLL(str(path))
    for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
        try:
            fn = getattr(lib,name)
        except AttributeError:
            continue
        fn.restype = ctypes.c_int
        info.append({"library":path.name,"symbol":name,"actual_threads":fn()})
        break
print(json.dumps(info))

