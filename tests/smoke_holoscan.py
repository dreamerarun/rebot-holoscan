"""Holoscan smoke test: 500 GPU messages through a 2-operator graph on the GB10."""
import time
import cupy as cp
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.conditions import CountCondition, PeriodicCondition


class Src(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("q")

    def compute(self, op_input, op_output, context):
        op_output.emit(cp.arange(7, dtype=cp.float32), "q")


class Sink(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("q")

    def initialize(self):
        self.n = 0
        self.t0 = time.time()

    def compute(self, op_input, op_output, context):
        q = op_input.receive("q")
        self.n += 1
        if self.n == 500:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            print(f"SMOKE OK: 500 msgs in {time.time() - self.t0:.2f}s, "
                  f"gpu sum={float(cp.sum(q * 2))}, device={name}")


class App(Application):
    def compose(self):
        src = Src(self, CountCondition(self, 500),
                  PeriodicCondition(self, recess_period=0.002), name="src")
        self.add_flow(src, Sink(self, name="sink"))


if __name__ == "__main__":
    App().run()
