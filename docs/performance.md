# Performance

[README](../README.md) · [Validation](validation.md) · [Study](../study/README.md)

PyMC-Marketing fits the Kettle & Pine model about three times faster than Foerster Marketing.
The numbers below come from the same study run as [validation.md](validation.md) ([process receipt](../results/kettle-and-pine/process-staggered-co-scheduled-calibrated.json)), on a shared Linux workstation with Java 25 (GraalVM), Foerster 0.1.137, Raster 0.2.1720, PyMC-Marketing 1.3.0 and PyMC 6.3.2.
Both engines run four chains of 2,000 warm-up and 4,000 retained draws.
These are single runs on a machine that was also doing other work, so treat differences of 10–20% as noise.

## Fitting

| Case | Foerster process (wall s) | of which gradient preparation | of which sampling | PyMC process (wall s) | PyMC `fit` call |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staggered | 660 | 73 | 523 | 213 | 202 |
| Overlapping | 743 | 75 | 587 | 237 | 232 |
| Overlapping + lift test | 896 | 169 | 645 | 265 | 258 |

| Case | Foerster CPU s | PyMC CPU s | Foerster min bulk ESS per CPU s | PyMC min bulk ESS per CPU s | Foerster max RSS | PyMC max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Staggered | 1,526 | 391 | 5.3 | 15.1 | 1.5 GB | 0.4 GB |
| Overlapping | 1,599 | 450 | 4.7 | 16.8 | 1.5 GB | 0.4 GB |
| Overlapping + lift test | 1,911 | 501 | 3.7 | 13.6 | 1.5 GB | 0.4 GB |

- **Process times** include JVM or Python startup. Foerster's include compiling the model's density and gradient (gradient preparation); PyMC used its persistent compiler cache, so its numbers include little compilation.
- **Sampling.** Foerster samples four chains on a two-thread executor. PyMC used its default NUTS sampler (`nuts_sampler="pymc"`) with two worker processes. Faster PyMC backends such as nutpie or NumPyro were not measured and would widen the gap.
- **Memory.** Foerster's JVM runs with a 512 MiB heap limit; resident memory is higher because of compiled code, metaspace and native buffers.

## Other stages

| Stage | Foerster (wall s) | PyMC (wall s) |
| --- | ---: | ---: |
| Density check at three points, including compilation | 133–208 | 7–15 |
| Decision process: allocate and compare plans | 89–124 (of which 34–47 for the decision itself) | 7–15 |
| Warehouse decision: SMC capacity inference and profit simulation for 7 budgets in 2 warehouse states | 222 | n/a |

## Where the time goes

Most of Foerster's sampling time is spent in the compiled gradient's backward sweep.
Raster 0.2.1720 splits that sweep into methods small enough for the JVM's JIT compiler; before that fix, the lift-test model ran interpreted and took more than 20 times longer per gradient.
Profiling points to two further sources of overhead: values boxed as objects between generated helper methods, and running inference inside Foerster's forkable execution context.
Both are being worked on in Raster and Spindel, and Foerster Marketing will benefit without changes of its own.
