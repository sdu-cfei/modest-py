"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import unittest
import tempfile
import os
from utilities.delete_logs import delete_logs


class TestDeleteLogs(unittest.TestCase):
    def setUp(self):
        # Temp directory
        self.temp_dir = tempfile.mkdtemp()
        # Temp log file
        self.log_path = os.path.join(self.temp_dir, 'test.log')
        log_file = open(self.log_path, 'w')
        log_file.close()

    def tearDown(self):
        try:
            os.remove(self.log_path)
        except WindowsError as e:
            pass # File already removed
        os.rmdir(self.temp_dir)

    def test_delete_logs(self):
        delete_logs(self.temp_dir)
        content = os.listdir(self.temp_dir)
        self.assertEqual(len(content), 0)


if __name__ == "__main__":
    unittest.main()