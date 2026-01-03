import os
import unittest

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))  # test_all.py 所在的 test/ 目录
    suite = unittest.defaultTestLoader.discover(
        start_dir=here, 
        pattern="test*.py",
        # top_level_dir=os.path.dirname(here)  # project 根目录
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)