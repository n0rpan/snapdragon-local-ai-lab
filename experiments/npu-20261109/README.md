# NPU operator measurement sources

Experimental sources from the 9 November measurement wave. See
[the report](../../docs/NPU-MEASUREMENTS.md) for timing semantics, exact binary
identities, retained failures and limits. These programs make real NPU calls;
they are not part of bootstrap or production.

Dependencies: the matching b12184 ggml headers/libs and v73 skel, the reviewed
FastRPC compatibility library, DSP shell/runtime dependencies, and a C++17
compiler. No vendor binaries or model weights are included. The operator
programs use the exported Hexagon registry directly; this prebuilt does not
export the dynamic-plugin entry point.

Run only through the existing NPU guard/supervisor under the inference lease,
with exactly one session, explicit MBUF=128M,128M,1G, VMEM unset, and temporary
heap access restored afterwards. Stop on the first device/mapping/kernel error;
never retry that boot, unload fastrpc, or read DMA-heap debugfs. Do not run these
binaries bare. The real launch adapters are kept in the private lab notes
(scripts/npu-stage1), not in this repository.

`npu_micro.cpp` covers shape/type sanity, dispatch-inclusive matmul timing and
independent SCALE graphs. `npu_prefill_ceiling.cpp` separates build, allocation,
upload, async submission, synchronization and DSP profile groups, fitting
wall time over 1/4/8/16 materialized matmuls. Largest N=2048 dense graphs use
only 1/4/8 nodes to preserve the 1 GiB session budget. The optional CLI argument
runs only the three largest shapes for stronger sampled numerical checks.
