import time, json
import numpy as np

import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def kahan_sum(x):
    s = 0.0
    c = 0.0
    for v in x:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def main():
    # 版本与设备
    env = {
        "numpy": np.__version__,
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "dtype": str(jnp.array(1.0).dtype),
    }

    # 和的数值稳定性
    x = np.ones(10_000_000, dtype=np.float64) * 1e-8
    t0 = time.time()
    s_np = x.sum()
    t1 = time.time()
    s_kahan = kahan_sum(x)
    t2 = time.time()

    # 向量化 + 设备
    s_jax = jnp.asarray(x).sum().item()

    result = {
        "env": env,
        "sum_numpy": float(s_np),
        "sum_kahan": float(s_kahan),
        "sum_jax": float(s_jax),
        "timing_sec": {
            "numpy_sum": t1 - t0,
            "kahan": t2 - t1,
        },
    }
    with open("smoke_test.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("OK. Wrote smoke_test.json")


if __name__ == "__main__":
    main()
