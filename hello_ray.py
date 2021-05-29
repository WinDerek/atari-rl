import ray
import time


ray.init(dashboard_host="0.0.0.0")


@ray.remote
def f(x):
    time.sleep(60)
    return x * x


futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))
