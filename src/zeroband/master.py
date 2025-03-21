from pccl import MasterNode
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='PCCL Master Node')
    parser.add_argument(
        '--listen-address',
        type=str,
        default='0.0.0.0:48148',
        help='Address for the master node to listen on (format: host:port)'
    )

    args = parser.parse_args()

    logging.info(f"Starting master node on {args.listen_address}")
    master: MasterNode = MasterNode(listen_address=args.listen_address)
    master.run()


if __name__ == '__main__':
    main()
