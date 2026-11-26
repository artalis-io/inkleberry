#!/usr/bin/env python3
"""Independent big-integer oracle for overflow-safe QPC tick conversion."""
import random
import subprocess
import sys
r = random.Random(0x1F123)
maximum = (1 << 64) - 1
cases = [(0, 0), (0, 1), (maximum, 1), (maximum, maximum),
         (maximum - 1, maximum), (maximum, 10_000_000)]
cases += [(r.getrandbits(64), r.getrandbits(64)) for _ in range(150)]
for ticks, frequency in cases:
    expected = ticks * 1_000_000_000 // frequency if frequency else maximum + 1
    actual = subprocess.check_output([sys.argv[1], str(ticks), str(frequency)], text=True).strip()
    assert actual == (str(expected) if expected <= maximum else 'overflow'), (ticks, frequency, actual, expected)
subprocess.run([sys.argv[1]], check=True)
print(f'{len(cases)} tick-conversion cases matched Python integer oracle')
