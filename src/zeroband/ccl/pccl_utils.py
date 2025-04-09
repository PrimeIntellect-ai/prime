from pccl import Communicator, torch, ReduceOp, ReduceOpDescriptor, ReduceDescriptor, ReduceOperandDescriptor, \
    DistributionHint, DataType, QuantizationAlgorithm, PCCLError, QuantizationOptions


def all_reduce_multiple_with_retry(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 128):
    descriptors = []
    tag = 0
    for tensor in tensors:
        reduce_op_descriptor = ReduceOpDescriptor.from_torch(
            send=tensor,
            recv=tensor,
            reduce_descriptor=ReduceDescriptor(
                count=tensor.numel(),
                op=op,
                tag=tag,
                operand_descriptor=ReduceOperandDescriptor(
                    datatype=DataType.FLOAT,
                    distribution_hint=DistributionHint.NORMAL
                ),
                quantization_options=QuantizationOptions(
                    quantized_datatype=DataType.FLOAT,
                    algorithm=QuantizationAlgorithm.NONE
                )
            )
        )
        descriptors.append(reduce_op_descriptor)
        tag += 1
    try:
        info = communicator.all_reduce_multiple_with_retry(descriptors, max_in_flight=max_in_flight)
        return True, info.tx_bytes, info.rx_bytes
    except PCCLError:
        return False, 0, 0
