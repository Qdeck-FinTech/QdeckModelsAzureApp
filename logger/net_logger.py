import logging

from Mercury import PythonLoggerAdapter

# Set up Python logger
py_logger = logging.getLogger("PythonLogger")
py_logger.setLevel(logging.DEBUG)


class PythonLogger:
    def Log(self, level, msg):
        py_logger.log(level * 10, msg)


net_logger = PythonLoggerAdapter(PythonLogger())
