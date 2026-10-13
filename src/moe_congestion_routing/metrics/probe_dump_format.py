"""The on-disk conventions a probe dump is written in, shared by the writer and every reader.

These live apart from ``router_probe``, which writes the dumps, because that module imports
``torch`` and the analysis side does not: reading a dump must work on a login node or a laptop
with no CUDA, no process group and no torch installed at all. A reader importing the writer for
one string would have pulled torch in behind it and quietly broken that.

The pack and the unpack must move together, which is the reason the bit order is a named constant
rather than a literal at each of the two call sites.
"""

# Every array is written in this row order: row n = sequence * seq_length + position, counting
# every probed sequence across every microbatch in probe-batch order. The router's own flatten is
# row n = position * micro_batch_size + sequence within a microbatch, so ProbeCapture.record
# changes the ordering, which is what makes dumps comparable across microbatch sizes.
TOKEN_AXIS_CONVENTION = "sequence-major: row n = sequence * seq_length + position"

# numpy's default made explicit: if num_experts is not divisible by 8 the routing_map array is
# padded with zeros, and bitorder `big` controls that these bits appear on the right for
# big-endian.
ROUTING_MAP_BITORDER = "big"
