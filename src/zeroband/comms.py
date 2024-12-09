import ipaddress
import os
import threading

import pccl

PCCL_INITIALIZED = False


class PcclCommunicator:

    def __init__(self, master_ip: str, master_port: int):
        global PCCL_INITIALIZED
        if not PCCL_INITIALIZED:
            pccl.pccl_init()
            PCCL_INITIALIZED = True

        if os.getenv("NO_IMPLICIT_MASTER") is None:
            self.master_handle = pccl.pccl_create_master(use_ipv4=True)
            self.master_thread = threading.Thread(target=self.run_master_blocking)
            self.master_thread.start()
        else:
            self.master_handle = 0
            self.master_thread = None

        # parse ip
        ip = ipaddress.ip_address(master_ip)

        if isinstance(ip, ipaddress.IPv4Address):
            is_ipv4 = True
        elif isinstance(ip, ipaddress.IPv6Address):
            is_ipv4 = False
        else:
            raise RuntimeError(f"Invalid master ip: {master_ip}")

        self.communicator = pccl.pccl_create_communicator()
        pccl.pccl_connect_master(self.communicator,
                                 pccl.ccoip_inet_protocol_t.inetIPv4 if is_ipv4 else pccl.ccoip_inet_protocol_t.inetIPv6,
                                 list(ip.packed), master_port)

    def run_master_blocking(self):
        try:
            pccl.pccl_run_master(self.master_handle)
        except RuntimeError:
            pccl.pccl_destroy_master(self.master_handle)
            self.master_handle = 0

    def interrupt_master(self):
        if self.master_handle != 0:
            pccl.pccl_interrupt_master(self.master_handle)
            self.master_thread.join()
            pccl.pccl_destroy_master(self.master_handle)

    def all_reduce(self, data_ptr, n_elements: int):
        reduce_info = pccl.pcclReduceInfo_t()
        try:
            pccl.pccl_allreduce(data_ptr, data_ptr, n_elements, pccl.pcclDataType_t.pcclFloat, pccl.pcclRedOp_t.pcclAvg,
                                1,
                                self.communicator, reduce_info)
        except Exception as e:
            print("all_reduce failed", e)

    def destroy(self):
        pccl.pccl_destroy_communicator(self.communicator)
        self.interrupt_master()
