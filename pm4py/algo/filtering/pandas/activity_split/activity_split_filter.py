from enum import Enum
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd

from pm4py.util import exec_utils, constants, xes_constants


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    SUBCASE_CONCAT_STR = "subcase_concat_str"
    CUT_MODE = "cut_mode"


def apply(df: pd.DataFrame, activity: str,
          parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> pd.DataFrame:
    """
    Splits the cases of a log (Pandas dataframe) into subcases based on the provision of an activity.
    There are as many subcases as many occurrences of a given activity occur.

    Example:
    Original log:

    [['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 'examine thoroughly',
    'check ticket', 'decide', 'pay compensation'],
    ['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 'check ticket',
    'examine casually', 'decide', 'reinitiate request', 'examine casually', 'check ticket', 'decide', 'reject request']]


    Log filtered using the activity split filter on 'reinitiate request' with cut_mode='this':

    [['register request', 'examine casually', 'check ticket', 'decide'],
    ['reinitiate request', 'examine thoroughly', 'check ticket', 'decide', 'pay compensation'],
    ['register request', 'examine casually', 'check ticket', 'decide'],
    ['reinitiate request', 'check ticket', 'examine casually', 'decide'],
    ['reinitiate request', 'examine casually', 'check ticket', 'decide', 'reject request']]


    Log filtered using the activity split filter on 'reinitiate request' with cut_mode='next':

    [['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request'],
    ['examine thoroughly', 'check ticket', 'decide', 'pay compensation'],
    ['register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request'],
    ['check ticket', 'examine casually', 'decide', 'reinitiate request'],
    ['examine casually', 'check ticket', 'decide', 'reject request']]


    Parameters
    ----------------
    df
        Dataframe
    activity
        Activity (splitter)
    parameters
        Parameters of the algorithm, including:
        - Parameters.ACTIVITY_KEY => activity key
        - Parameters.CASE_ID_KEY => case id
        - Parameters.SUBCASE_CONCAT_STR => concatenator between the case id and the subtrace index in the filtered df
        - Parameters.CUT_MODE => mode of cut:
            - "this" means that an event with the specified activity goes to the next subcase
            - "next" means that the following event (to the given activity) goes to the next subcase.

    Returns
    ----------------
    filtered_df
        Dataframe in which the cases are split into subcases

    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME)
    subcase_concat_str = exec_utils.get_param_value(Parameters.SUBCASE_CONCAT_STR, parameters, "##@@")
    cut_mode = exec_utils.get_param_value(Parameters.CUT_MODE, parameters, "this")

    df = df.copy()
    cases = df[case_id_key].to_numpy()
    activities = df[activity_key].to_numpy()
    c_unq, c_ind, c_counts = np.unique(cases, return_index=True, return_counts=True)
    res = []

    i = 0
    while i < len(c_unq):
        rel_count = 0
        this_case = str(c_unq[i]) + subcase_concat_str + str(rel_count)

        j = 0
        while j < c_counts[i]:
            if activities[c_ind[i] + j] == activity:
                rel_count += 1
                next_case = str(c_unq[i]) + subcase_concat_str + str(rel_count)
                if cut_mode == "this":
                    res.append(next_case)
                else:
                    res.append(this_case)
                this_case = next_case
            else:
                res.append(this_case)
            j = j + 1
        i = i + 1

    df[case_id_key] = res

    return df