import os
import random
import string
import sys
import unittest

sys.path.insert(0, os.environ["PROJECT"])
from logfmt import format, parse  # noqa: E402

KEY_CHARS = string.ascii_letters + string.digits + '_.-'
VALUE_CHARS = 'ab Z09_-.=\\"\n\t é日ü\'#'


def random_record(rng):
    record = {}
    for _ in range(rng.randint(0, 6)):
        key = ''.join(rng.choice(KEY_CHARS) for _ in range(rng.randint(1, 8)))
        if rng.random() < 0.15:
            record[key] = True
        else:
            record[key] = ''.join(rng.choice(VALUE_CHARS) for _ in range(rng.randint(0, 12)))
    return record


class RoundTripProperty(unittest.TestCase):
    def test_500_seeded_records(self):
        rng = random.Random(20240611)
        for i in range(500):
            record = random_record(rng)
            line = format(record)
            self.assertNotIn('\n', line, (i, record))
            self.assertEqual(parse(line), record, (i, record, line))


if __name__ == '__main__':
    unittest.main()
