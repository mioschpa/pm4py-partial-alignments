from enum import Enum
from statistics import mean

from pm4py.util import exec_utils, constants, xes_constants
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util.business_hours import BusinessHours


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    BUSINESS_HOURS = "business_hours"
    WORKTIMING = "worktiming"
    WEEKENDS = "weekends"


DIFF_KEY = "@@diff"


def apply(log, parameters=None):
    """
    Gets the sojourn time per activity on an event log object

    Parameters
    --------------
    dataframe
        Pandas dataframe
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => activity key
        - Parameters.START_TIMESTAMP_KEY => start timestamp key
        - Parameters.TIMESTAMP_KEY => timestamp key
        - Parameters.BUSINESS_HOURS => calculates the difference of time based on the business hours, not the total time.
                                        Default: False
        - Parameters.WORKTIMING => work schedule of the company (provided as a list where the first number is the start
            of the work time, and the second number is the end of the work time), if business hours are enabled
                                        Default: [7, 17] (work shift from 07:00 to 17:00)
        - Parameters.WEEKENDS => indexes of the days of the week that are weekend
                                        Default: [6, 7] (weekends are Saturday and Sunday)

    Returns
    --------------
    soj_time_dict
        Sojourn time dictionary
    """
    if parameters is None:
        parameters = {}

    business_hours = exec_utils.get_param_value(Parameters.BUSINESS_HOURS, parameters, False)
    worktiming = exec_utils.get_param_value(Parameters.WORKTIMING, parameters, [7, 17])
    weekends = exec_utils.get_param_value(Parameters.WEEKENDS, parameters, [6, 7])

    log = log_converter.apply(log, parameters=parameters)

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    start_timestamp_key = exec_utils.get_param_value(Parameters.START_TIMESTAMP_KEY, parameters,
                                                     xes_constants.DEFAULT_TIMESTAMP_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)

    durations_dict = {}
    activities = [ev[activity_key] for trace in log for ev in trace]
    for act in activities:
        durations_dict[act] = []

    for trace in log:
        for event in trace:
            activity = event[activity_key]
            if business_hours:
                bh = BusinessHours(event[start_timestamp_key].replace(tzinfo=None), event[timestamp_key].replace(tzinfo=None),
                                   worktiming=worktiming,
                                   weekends=weekends)
                durations_dict[activity].append(bh.getseconds())
            else:
                start_time = event[start_timestamp_key].timestamp()
                complete_time = event[timestamp_key].timestamp()
                durations_dict[activity].append(complete_time - start_time)

    for act in durations_dict:
        durations_dict[act] = mean(durations_dict[act])

    return durations_dict
