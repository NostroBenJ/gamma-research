"""Signal research: measure first, model later.

The CISD project ran the other way round -- a large model built to completion,
then measured, and it failed with t = -3.54. The measurement was cheap; the
model was expensive. This package inverts that. A signal is a function from
(bars, index) to a direction, and anything that fits that shape can be scored
in seconds against the same harness and the same null.
"""
