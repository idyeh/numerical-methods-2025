import sys, platform, json
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

out = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "jax": jax.__version__,
    "devices": [str(d) for d in jax.devices()],
    "dtype_check": str(jnp.array(1.0).dtype),
}
print(json.dumps(out, indent=2, ensure_ascii=False))
