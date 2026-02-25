#!/usr/bin/env python3
import sys
import struct
import time
import argparse
import numpy as np

import crypto_lh as crypto

maximum = 2 ** 32 - 1
chunk_size = 8192
w = 14

generator = None
debug = False


def output(expected):
    """
    Outputs numbers to stdout
    :return: the next seed
    """
    numbers = generator.generate_chunk(expected, debug)

    if len(numbers) != expected:
        print(f"[WARN] Expected {expected}, got {len(numbers)}", file=sys.stderr)
        raise SystemExit(1)

    sys.stdout.buffer.write(struct.pack('<{}I'.format(expected), *numbers))

    if debug:
        for num in numbers:
            print(num, file=sys.stderr)
    sys.stdout.flush()


def pipe():
    """
    Main function to run the PIPE interface for testing.
    """
    print(f"--- Testing Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    start_time = time.time()

    try:
        while True:
            output(chunk_size)

            chunks_sent += 1
            if chunks_sent % 4000 == 0:
                elapsed = time.time() - start_time
                rate = (chunks_sent * chunk_size) / elapsed
                print(f"[INFO] Sent {chunks_sent:,} chunks "
                      f"({chunks_sent * chunk_size:,} numbers) "
                      f"at {rate:,.0f} numbers/sec", file=sys.stderr)
                sys.stderr.flush()

    except BrokenPipeError:
        print("\n--- Stream closed by Tester. Exiting gracefully. ---", file=sys.stderr)
        sys.stderr.flush()
    except KeyboardInterrupt:
        print("\n--- Stream interrupted by user. Exiting. ---", file=sys.stderr)
        sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(description="Crypto Testing Interface.")
    parser.add_argument("seeds", type=int, nargs='+',
                        help="One 64-bit integer (will be expanded) or exactly five 64-bit integers.")
    parser.add_argument("--delta", type=int, default=0, help="delta (step size)")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    args = parser.parse_args()

    global generator, debug
    debug = args.debug

    raw_seeds = args.seeds
    num_seeds = len(raw_seeds)

    states = np.zeros(5, dtype=np.uint64)

    if num_seeds == 1:
        # simple SplitMix64 expansion to get 5 uncorrelated starting states
        base_seed = raw_seeds[0]
        print(f"[INFO] Expansion: One seed provided. Generating 5 states from {base_seed}...", file=sys.stderr)

        current = base_seed
        for i in range(5):
            current = (current + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
            z = current
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
            z = (z & 0xFFFFFFFFFFFFFFFF)
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb
            z = (z & 0xFFFFFFFFFFFFFFFF)
            states[i] = z ^ (z >> 31)
            print(f"[INFO] States[{i}] = {states[i]}...", file=sys.stderr)

    elif num_seeds == 5:
        print(f"[INFO] Direct: 5 seeds accepted.", file=sys.stderr)
        for seed in range(len(raw_seeds)):
            print(f"[INFO] Seed[{seed}]: {raw_seeds[seed]}", file=sys.stderr)
        states[:] = raw_seeds
    else:
        parser.error(f"You must provide either 1 seed or exactly 5 seeds. You provided {num_seeds}.")

    generator = crypto.CryptoLehmer(states, w, args.delta, 0, maximum)

    pipe()


if __name__ == "__main__":
    main()
