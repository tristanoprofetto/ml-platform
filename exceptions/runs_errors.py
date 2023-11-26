class ParallelRunsError(Exception):
    """Exception raised for errors in the parallel_runs function.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class RegisterError(Exception):
    """Exception raised for errors in the register function.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class RunExperimentError(Exception):
    """Exception raised for errors in the run_experiment function.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class GetParamsError(Exception):
    """Exception raised for errors in the get_params function.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message


class GetArgsError(Exception):
    """Exception raised for errors in the get_args function.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message

        



