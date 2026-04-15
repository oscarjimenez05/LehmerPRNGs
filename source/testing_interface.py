#!/usr/bin/env python3
import sys
import struct
import time
import argparse

import c_lcg_lh as lcg
import xor_lh as xor

import alternatives.lcg_fenwick as lfw
import alternatives.xor_fenwick as xfw
import alternatives.logistic_lh as log
import alternatives.gaussian_lh as gau
import alternatives.slope_lh as slp
import alternatives.decay_lh as dec
import alternatives.comp as comp

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


def file(total_numbers):
    """
    Generate a fixed number of random 32-bit unsigned integers and write to stdout as binary.
    """
    print(f"--- Testing Fixed Output Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Total numbers to generate = {total_numbers:,}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    numbers_sent = 0
    start_time = time.time()

    try:
        while numbers_sent < total_numbers:
            remaining = total_numbers - numbers_sent
            current_chunk = min(chunk_size, remaining)

            output(current_chunk)

            chunks_sent += 1
            numbers_sent += current_chunk

            if chunks_sent % 500 == 0:
                elapsed = time.time() - start_time
                rate = numbers_sent / elapsed
                print(f"[INFO] {round(100*numbers_sent/total_numbers, 3)}% "
                      f"Sent {numbers_sent:,}/{total_numbers:,} numbers "
                      f"({rate:,.0f} nums/sec)", file=sys.stderr)
                sys.stderr.flush()

        sys.stdout.flush()
        print(f"--- Completed {numbers_sent:,} numbers. ---", file=sys.stderr)

    except BrokenPipeError:
        print("\n--- Stream closed early. Exiting gracefully. ---", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n--- Interrupted by user. Exiting. ---", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Testing Interface.")
    parser.add_argument("mode", choices=['f', 'p'], help="(f)ile or (p)ipe.")
    parser.add_argument("seed", type=int, help="seed")
    parser.add_argument("delta", type=int, help="delta")

    parser.add_argument("--total", type=int, help="total numbers to generate (required for file mode)")
    parser.add_argument("--algo", choices=['lcg', 'xor', 'lfw', 'xfw', 'log',
                                           'gau', 'slp', 'dec', 'Lcomp', 'Hcomp'], default='lcg', help="Choose generator algorithm")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    args = parser.parse_args()

    if args.mode == 'f' and args.total is None:
        parser.error("the 'f' mode requires --total <number>.")

    global generator, debug
    debug = args.debug

    match args.algo:
        case 'lcg':
            generator = lcg.LcgLehmer(args.seed, w, args.delta, 0, maximum)
        case 'xor':
            generator = xor.XorLehmer(args.seed, w, args.delta, 0, maximum)
        case 'lfw':
            generator = lfw.LcgFenwick(args.seed, w, args.delta, 0, maximum)
        case 'xfw':
            generator = xfw.XorFenwick(args.seed, w, args.delta, 0, maximum)
        case 'log':
            generator = log.LogisticLehmer(args.seed, w, args.delta, 0, maximum)
        case 'gau':
            generator = gau.GaussianLehmer(args.seed, w, args.delta, 0, maximum)
        case 'slp':
            generator = slp.SlopeLehmer(args.seed, w, args.delta, 0, maximum)
        case 'dec':
            generator = dec.DecayLehmer(args.seed, w, args.delta, 0, maximum)
        case 'Lcomp':
            generator = comp.LmLcg(args.seed, 0, maximum)
        case 'Hcomp':
            generator = comp.HighBitsLcg(args.seed, 0, maximum)

    # -----------------------------------------------

    if args.mode == 'f':
        file(args.total)
    elif args.mode == 'p':
        pipe()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
