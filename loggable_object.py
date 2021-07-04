from utils.general_utils import log_print


class LoggableObject:
    """ The top class from which all other classes inherit. """

    def __init__(self, indent):
        self.indent = indent

    def increment_indent(self):
        self.indent += 1

    def decrement_indent(self):
        self.indent -= 1

    def log_print(self, my_str):
        log_print(self.__class__.__name__, self.indent, my_str)
