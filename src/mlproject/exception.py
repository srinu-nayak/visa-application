import traceback
import sys

def get_error_message(error):
    tb = traceback.extract_tb(sys.exc_info()[2])[-1]  # get last traceback
    file_name, line_number, _, _ = tb
    return f"Error in [{file_name}], line [{line_number}]: {error}"

class CustomException(Exception):
    def __init__(self, error):
        super().__init__(get_error_message(error))
