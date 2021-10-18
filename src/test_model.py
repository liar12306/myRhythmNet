import config
import os
if os.path.exists(config.test):
    print("exit")
else:
    print("not exit")