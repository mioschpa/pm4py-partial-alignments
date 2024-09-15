'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import random

import pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star
from pm4py.visualization.petri_net import visualizer

"""
This module contains code that allows us to compute alignments on the basis of a regular A* search on the state-space
of the synchronous product net of a trace and a Petri net.
The main algorithm follows [1]_.
When running the log-based variant, the code is running in parallel on a trace based level.
Furthermore, by default, the code applies heuristic estimation, and prefers those states that have the smallest h-value
in case the f-value of two states is equal.

References
----------
.. [1] Sebastiaan J. van Zelst et al., "Tuning Alignment Computation: An Experimental Evaluation",
      ATAED@Petri Nets/ACSD 2017: 6-20. `http://ceur-ws.org/Vol-1847/paper01.pdf`_.

"""
import heapq
import sys
import time
from copy import copy
from enum import Enum

import numpy as np

from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset, get_transition_by_name, add_arc_from_to, remove_place, pre_set, post_set
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union, Tuple
from pm4py.objects.log.obj import EventLog, EventStream, Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking, EventNet
from pm4py.util import typing
import pandas as pd


class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = 'trace_cost_function'
    PARAM_MODEL_COST_FUNCTION = 'model_cost_function'
    PARAM_SYNC_COST_FUNCTION = 'sync_cost_function'
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = 'ret_tuple_as_trans_desc'
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    VARIANTS_IDX = "variants_idx"
    RETURN_SYNC_COST_FUNCTION = "return_sync_cost_function"


PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value


def get_best_worst_cost(petri_net, initial_marking, final_marking, parameters=None):
    """
    Gets the best worst cost of an alignment

    Parameters
    -----------
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking

    Returns
    -----------
    best_worst_cost
        Best worst cost of alignment
    """
    if parameters is None:
        parameters = {}
    trace = log_implementation.Trace()

    best_worst = pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.apply(trace, petri_net,
                                                                                                  initial_marking,
                                                                                                  final_marking,
                                                                                                  parameters=parameters)

    if best_worst is not None:
        return best_worst['cost']

    return None


def __get_im_fm(trace_net: PetriNet):
    p_list = list()
    im = Marking()
    fm = Marking()

    #  initial marking
    for p in trace_net.places:
        if len(p.in_arcs) == 0:
            p_list.append(p)

    for i in range(len(p_list)):
        im[p_list[i]] = 1

    # final marking
    p_list = list()
    for p in trace_net.places:
        if len(p.out_arcs) == 0:
            p_list.append(p)

    for i in range(len(p_list)):
        fm[p_list[i]] = 1

    return im, fm


def apply(trace_net: PetriNet, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking,
          parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> PetriNet:
    """
    Performs the basic alignment search, given a trace_net and a model petri_net.

    Parameters
    ----------
    trace_net: :class:`pm4py.objects.petri.net.PetriNet` the partial trace represented as a Petri net
    petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
    parameters: :class:`dict` (optional) dictionary containing one of the following:
        Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
        Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        model cost
        Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        synchronous costs
        Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events

    Returns
    -------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)
    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    trace_net_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_CONSTR_FUNCTION, parameters,
                                                           None)
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
                                                                      parameters, construct_trace_net_cost_aware)

    if trace_cost_function is None:
        # trace_cost_function = list(
        #     map(lambda e: utils.STD_MODEL_LOG_MOVE_COST, trace_net))
        # for partial traces: create a list of length len(trace_net.places). fill list with LOG_MOVE_COST
        trace_cost_function = list()
        for _ in trace_net.places:
            trace_cost_function.append(utils.STD_MODEL_LOG_MOVE_COST)
        parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    if model_cost_function is None:
        # reset variables value
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in petri_net.transitions:
            if t.label is not None:
                model_cost_function[t] = utils.STD_MODEL_LOG_MOVE_COST
                sync_cost_function[t] = utils.STD_SYNC_COST
            else:
                model_cost_function[t] = utils.STD_TAU_COST
        parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
        parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function

    # Partial Traces do not need a conversion as they are already Petri Nets
    # if trace_net_constr_function is not None:
    #     # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old version
    #     trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
    # else:
    #     trace_net, trace_im, trace_fm, parameters[
    #         Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
    #                                                                                  trace_cost_function,
    #                                                                                  activity_key=activity_key)

    trace_im, trace_fm = __get_im_fm(trace_net)

    # Sync Product must distinguish between model activities and trace activities (maybe via act. names?)
    alignment, sync_prod = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm,
                                           parameters)
    #visualizer.apply(sync_prod).view()
    opt_ali_net = construct_optimal_alignment_net(sync_prod, alignment)
    #visualizer.apply(opt_ali_net).view()
    if not __check_valid_opt_ali_net(opt_ali_net):
        repair_opt_ali_net(opt_ali_net, alignment["alignment"])
    opt_p_alignment = construct_opt_p_ali(opt_ali_net)
    return opt_p_alignment


def __check_valid_opt_ali_net(opt_ali_net: PetriNet):
    all_places = opt_ali_net.places
    for place in all_places:
        if len(place.in_arcs) > 1 or len(place.out_arcs) > 1:
            return False
    return True


def __find_transition_by_label(petri_net: PetriNet, alignment: list[str], label: str, bad_places: set[PetriNet.Place],
                               second_call=False) -> PetriNet.Transition:
    # if only one transition has this label, then return it
    possible_transitions = set()
    for t in petri_net.transitions:
        if t.label == label:
            possible_transitions.add(t)
    if len(possible_transitions) == 1:
        return possible_transitions.pop()
    # put a token in each starting place (the places without inc arcs)
    curr_marking = {}
    for p in petri_net.places:
        if not p.in_arcs:
            curr_marking[p] = 1
        else:
            curr_marking[p] = 0
    # replay alignment - iterate over alignment
    for i in range(len(alignment)):
        # if it is the label
        if alignment[i] == label:
            # return the one transition that is activated
            for trans in possible_transitions:
                t_preset = pre_set(trans)
                for elem in t_preset:
                    if curr_marking[elem] == 0:
                        break
                else:
                    # if pre_set contains a bad_place
                    if not second_call:
                        return trans
                    if second_call and t_preset.intersection(bad_places):
                        return trans
        # get transition by alignment label
        next_transitions = set()
        for t in petri_net.transitions:
            if t.label == alignment[i]:
                next_transitions.add(t)
        # if two transitions found
        if len(next_transitions) == 1:
            next_transition = next_transitions.pop()
        else:
            # choose the activated one
            for trans in next_transitions:
                t_preset = pre_set(trans)
                for elem in t_preset:
                    if curr_marking[elem] == 0:
                        break
                else:
                    activated_trans = trans
                    break
            next_transition = activated_trans
        # fire alignment_i
        consume_tokens: set[PetriNet.Place] = set()
        for arc in next_transition.in_arcs:
            consume_from_place = arc.source
            curr_marking[consume_from_place] = curr_marking[consume_from_place] - 1
        for arc in next_transition.out_arcs:
            produce_to_place = arc.target
            curr_marking[produce_to_place] = curr_marking[produce_to_place] + 1


def __label_already_seen(alignment: list[str], k :int):
    # checks, if label on position k in the alignment already happened before
    label = alignment[k]
    for j in range(k):
        if alignment[j] == label:
            return True
    return False


def repair_opt_ali_net(opt_ali_net: PetriNet, alignment):
    # find bad_places
    bad_places = set()
    for place in opt_ali_net.places:
        if len(place.in_arcs) > 1 or len(place.out_arcs) > 1:
            bad_places.add(place)
    # as long as there are bad places
    while bad_places:
        # iterate through alignment
        for i in range(len(alignment)):
            if i == 0:
                prev_t = __find_transition_by_label(opt_ali_net, alignment, alignment[i], bad_places)
            if __label_already_seen(alignment, i+1):
                next_t = __find_transition_by_label(opt_ali_net, alignment, alignment[i+1], bad_places, second_call=True)
            else:
                next_t = __find_transition_by_label(opt_ali_net, alignment, alignment[i + 1], bad_places)
            bad_places_found = set()
            for out_arc in prev_t.out_arcs:
                if out_arc.target in bad_places:
                    set_next_t_inc_arcs = next_t.in_arcs
                    for inc_a in set_next_t_inc_arcs:
                        if out_arc.target == inc_a.source:  # to get exact arc of current alignment path
                            bad_places_found.add(out_arc.target) # TODO do you need a break statement after this?
            bad_place: PetriNet.Place = None
            # when bad place reached:
            if len(bad_places_found) > 0:
                bad_place = bad_places_found.pop()
                bad_places.remove(bad_place)
            else:
                prev_t = next_t
                continue
            # identify arc1 and arc2
            for t_out_arc in prev_t.out_arcs:
                if t_out_arc.target == bad_place:
                    arc1 = t_out_arc
                    break
            for t_in_arc in next_t.in_arcs:
                if t_in_arc.source == bad_place:
                    arc2 = t_in_arc
                    break
            # remove arc1 and arc2 from bad_place
            to_be_removed = None
            for possible_arc in bad_place.in_arcs:
                if possible_arc.source == arc1.source and possible_arc.target == arc1.target:
                    to_be_removed = possible_arc
            bad_place.in_arcs.remove(to_be_removed)
            to_be_removed = None
            for possible_arc in bad_place.out_arcs:
                if possible_arc.source == arc2.source and possible_arc.target == arc2.target:
                    to_be_removed = possible_arc
            bad_place.out_arcs.remove(to_be_removed)
            # remove arc1 and arc2 from net
            arc_to_remove1 = None
            arc_to_remove2 = None
            for a in opt_ali_net.arcs:
                if a.source == prev_t and a.target == bad_place:
                    arc_to_remove1 = a
                    continue
                if a.source == bad_place and a.target == next_t:
                    arc_to_remove2 = a
                if arc_to_remove1 is not None and arc_to_remove2 is not None:
                    break
            opt_ali_net.arcs.remove(arc_to_remove1)
            opt_ali_net.arcs.remove(arc_to_remove2)
            # remove arc1 from prev_trans
            prev_t.out_arcs.remove(arc1)
            # remove arc2 from next_trans
            next_t.in_arcs.remove(arc2)
            # create new place
            new_label1 = bad_place.name[0]
            new_label2 = bad_place.name[1]

            new_rand = str(random.randint(1, 1000))
            if new_label1.startswith(">>"):
                new_label2 = new_label2 + "_" + new_rand
            else:
                new_label1 = new_label1 + "_" + new_rand
            new_place = PetriNet.Place((new_label1, new_label2))
            opt_ali_net.places.add(new_place)
            # add_new_arc arc1 from prev_trans to new place
            add_arc_from_to(prev_t, new_place, opt_ali_net)
            # add_new_arc arc2 from new place to next_trans
            add_arc_from_to(new_place, next_t, opt_ali_net)
            #visualizer.apply(opt_ali_net).view()
            prev_t = next_t
            break
        # check if there are more (new) bad_places
        for place in opt_ali_net.places:
            if len(place.in_arcs) > 1 or len(place.out_arcs) > 1:
                bad_places.add(place)
    return


# DEPRECATED FUNCTION
def repair_sync_prod(sync_prod: PetriNet, alignment):
    # this method repairs the synchronous product by performing unfolding on some places that are on cycles
    # identify affected places (subsection between preset and postset for a single transition TODO this is only self-loops
    affected_places = set()
    for t in sync_prod.transitions:
        t_preset = pre_set(t)
        t_postset = post_set(t)
        for p in t_preset:
            if p in t_postset and p not in affected_places:
                affected_places.add(p)
    # for an affected place: identify transitions that loop over this place
    for p in affected_places:
        affected_transitions = set()
        p_preset = pre_set(p)
        p_postset = post_set(p)
        for t in p_preset:
            if t in p_postset and p not in affected_transitions:
                affected_transitions.add(t)
        # iterate over alignment as long as it is an affected transition:
        for i in range(len(alignment)):
            curr_ali_transition = __find_current_transition_from_alignment(sync_prod, i, alignment)
            if curr_ali_transition not in affected_transitions:
                continue
            # create a new place in preset of the current alignment transition with exactly one out_arc
            # only in_arcs that are NOT affected transitions retain as in_arcs
            new_inc_arcs = set()
            p_in_arcs = p.in_arcs
            for inc_t in p_in_arcs:
                if inc_t.source not in affected_transitions:
                    new_inc_arcs.add(inc_t)
            if p.name[0].startswith(">>"):
                new_name = (">>", "added_place_" + str(i))
            else:
                new_name = ("added_place_" + str(i), ">>")
            new_place = PetriNet.Place(new_name)
            new_place.ass_trans = set()
            sync_prod.places.add(new_place)
            for arc in new_inc_arcs:
                #  new_place.in_arcs.add(PetriNet.Arc(arc.source, new_place))
                add_arc_from_to(arc.source, new_place, sync_prod)
            # and exactly one out_arc
            new_arc = PetriNet.Arc(new_place, curr_ali_transition)
            new_place.out_arcs.add(new_arc)
            sync_prod.arcs.add(new_arc)
            # modify original place: delete arcs to previous transitions including curr_transition
            for prev_arc in new_inc_arcs:
                p.in_arcs.remove(prev_arc)
                sync_prod.arcs.remove(prev_arc)
            # special case: remove outgoing arc to curr_transition
            for arc in p.out_arcs:
                if arc.target == curr_ali_transition:
                    p.out_arcs.remove(arc)
                    sync_prod.arcs.add(arc)
                    break
            # modify each previous transition leading to original place: reroute arc to new place
            my_list = list()  # list of pairs of transition and according arc to be removed
            for arc in new_inc_arcs:  # finds previous transitions
                for out_arc in arc.source.out_arcs:  # reroutes arcs to newly created place
                    if out_arc.target == p:
                        my_list.append((arc.source, out_arc))
            for i in range(len(my_list)):
                temp_t = my_list[i][0]
                temp_a = my_list[i][1]
                temp_t.out_arcs.remove(temp_a)

                #  temp_arc_to_add = PetriNet.Arc(temp_t, new_place)
                #  temp_t.out_arcs.add(temp_arc_to_add)
                #  sync_prod.arcs.add(temp_arc_to_add)

            # special case: remove in_arc from current_transition
            arc_to_del = None
            for arc in curr_ali_transition.in_arcs:
                if arc.source == p:
                    arc_to_del = arc
                    break
            curr_ali_transition.in_arcs.remove(arc_to_del)
            sync_prod.arcs.remove(arc_to_del)
            temp_arc_to_add = PetriNet.Arc(new_place, curr_ali_transition)
            curr_ali_transition.in_arcs.add(temp_arc_to_add)
            affected_transitions.remove(curr_ali_transition)
    return


def __find_current_transition_from_alignment(sync_prod, pos, alignment):
    # this function finds the transition that corresponds to the label of the pos-element of the alignment
    # get transitions by label
    possible_transitions = get_transitions_by_label(alignment[pos], sync_prod)
    assert len(possible_transitions) != 0
    if len(possible_transitions) == 1:
        return list(possible_transitions)[0]
    # replay the alignment in the sync prod ang find out that transition is activated
    curr_marking = get_initial_marking(sync_prod)
    for i in range(pos):
        ali_transition = alignment[i]
        next_transition_set = get_transitions_by_label(ali_transition, sync_prod)
        if len(next_transition_set) == 1:
            next_transition = list(next_transition_set)[0]
        else:
            # retrieve only the one transition that is activated
            next_transition = find_activated_transition(curr_marking, next_transition_set)
        curr_marking = fire_transition(curr_marking, next_transition)
    # find the first activated one
    next_transition = find_activated_transition(curr_marking, possible_transitions)
    return next_transition


def check_self_loops(sync_prod):
    # if a place is preset as well as postset of a transition than a cycle is detected
    all_transitions = sync_prod.transitions
    for t in all_transitions:
        t_preset = pre_set(t)
        t_postset = post_set(t)
        for place in t_preset:
            if place in t_postset:
                return True
    return False


def get_initial_marking(sync_prod: PetriNet):
    initial_marked_places = set()
    for place in sync_prod.places:
        if len(place.in_arcs) == 0:
            initial_marked_places.add(place)
    return initial_marked_places


def get_transitions_by_label(ali_transition, sync_prod):
    found_transitions = set()
    for t in sync_prod.transitions:
        if t.label == ali_transition:
            found_transitions.add(t)
    return found_transitions


def find_activated_transition(curr_marking, transition_set):
    marked_place_names = set()
    for p in curr_marking:
        new_place = PetriNet.Place(p.name)
        marked_place_names.add(new_place.name)
    for t in transition_set:
        my_preset = pre_set(t)
        for place in my_preset:
            if place.name not in marked_place_names:
                break
        else:
            return t
    return None


def fire_transition(curr_marking, next_transition):
    my_preset = pre_set(next_transition)
    my_postset = post_set(next_transition)
    # remove tokens of preset
    for place in my_preset:
        curr_marking.remove(place)
    # add tokens to postset
    for place in my_postset:
        curr_marking.add(place)
    return curr_marking


def __find_exisiting_place(opt_net: PetriNet, place: PetriNet.Place):
    all_places = opt_net.places
    for p in all_places:
        if p.name == place.name:
            return p


def __arc_exists(source: PetriNet.Arc, target: PetriNet.Arc, net: PetriNet):
    all_arcs = net.arcs
    for a in all_arcs:
        if a.source == source:
            if a.target == target:
                return True
    return False


def __add_pre_set_of_transition(next_transition, opt_net):
    my_preset = pre_set(next_transition)
    places_to_add = set()
    arcs_to_add = set()
    for place in my_preset:
        out_arcs_set = place.out_arcs
        for outward_arc in out_arcs_set:
            #  only add subset of arcs that connect places that are still retained in opt net
            if outward_arc.target in opt_net.transitions:
                if place not in opt_net.places:
                    places_to_add.add(place)
                    arcs_to_add.add(PetriNet.Arc(place, next_transition))
                else:
                    temp_place = __find_exisiting_place(opt_net, place)
                    arcs_to_add.add(PetriNet.Arc(temp_place, next_transition))

    for p in places_to_add:
        opt_net.places.add(p)
    for a in arcs_to_add:
        if not __arc_exists(a.source, a.target, opt_net):
            add_arc_from_to(a.source, a.target, opt_net)


def __add_post_set_of_transition(next_transition, opt_net):
    my_postset = post_set(next_transition)
    places_to_add = set()
    arcs_to_add = set()
    for place in my_postset:
        in_arcs_set = place.in_arcs
        for inward_arc in in_arcs_set:
            #  only add subset of arcs that connect places that are still retained in opt net
            if inward_arc.source in opt_net.transitions:
                if place not in opt_net.places:
                    places_to_add.add(place)
                    arcs_to_add.add(PetriNet.Arc(next_transition, place))
                else:
                    temp_place = __find_exisiting_place(opt_net, place)
                    arcs_to_add.add(PetriNet.Arc(next_transition, temp_place))
    for p in places_to_add:
        opt_net.places.add(p)
    for a in arcs_to_add:
        if not __arc_exists(a.source, a.target, opt_net):
            add_arc_from_to(a.source, a.target, opt_net)


def __check_seen(seen_arcs: set[PetriNet.Arc], arc: PetriNet.Arc):
    # checks if seen_arcs contains arc
    arc_from = arc.source
    arc_to = arc.target
    for seen_arc in seen_arcs:
        if seen_arc.source == arc_from and seen_arc.target == arc_to:
            return True
    return False


def __remove_unconnected_and_duplicated_arcs(opt_net):
    p_set = opt_net.places
    t_set = opt_net.transitions

    # remove unused arcs from places
    for p in p_set:
        arcs_to_delete = set()
        for a in p.in_arcs:
            if a.source not in t_set:
                arcs_to_delete.add(a)
        [p.in_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        arcs_to_delete.clear()
        for a in p.out_arcs:
            if a.target not in t_set:
                arcs_to_delete.add(a)
        [p.out_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        # special case: remove unused references to some transitions
        transitions_to_delete = set()
        if p.ass_trans is not None:
            for t in p.ass_trans:
                if t not in t_set:
                    transitions_to_delete.add(t)
            [p.ass_trans.remove(tran) for tran in transitions_to_delete]

    # remove unused arcs from transitions
    for t in t_set:
        arcs_to_delete = set()
        for a in t.in_arcs:
            if a.source not in p_set:
                arcs_to_delete.add(a)
        [t.in_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        arcs_to_delete.clear()
        for a in t.out_arcs:
            if a.target not in p_set:
                arcs_to_delete.add(a)
        [t.out_arcs.remove(arc_del) for arc_del in arcs_to_delete]

    # remove duplicates
    arcs_to_delete = set()
    for p in p_set:
        seen_arcs = set()
        for arc in p.in_arcs:
            already_seen = __check_seen(seen_arcs, arc)
            if already_seen:
                arcs_to_delete.add(arc)
            else:
                seen_arcs.add(arc)
        [p.in_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        arcs_to_delete.clear()
        seen_arcs.clear()
        for arc in p.out_arcs:
            already_seen = __check_seen(seen_arcs, arc)
            if already_seen:
                arcs_to_delete.add(arc)
            else:
                seen_arcs.add(arc)
        [p.out_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        arcs_to_delete.clear()

    for t in t_set:
        seen_arcs = set()
        for arc in t.in_arcs:
            already_seen = __check_seen(seen_arcs, arc)
            if already_seen:
                arcs_to_delete.add(arc)
            else:
                seen_arcs.add(arc)
        [t.in_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        arcs_to_delete.clear()
        seen_arcs.clear()
        for arc in t.out_arcs:
            already_seen = __check_seen(seen_arcs, arc)
            if already_seen:
                arcs_to_delete.add(arc)
            else:
                seen_arcs.add(arc)
        [t.out_arcs.remove(arc_del) for arc_del in arcs_to_delete]
        arcs_to_delete.clear()


def construct_optimal_alignment_net(sync_prod: PetriNet, alignment_dict):
    # input is an alignment and a synchronous product
    opt_net = PetriNet("Optimal alignment net")
    t_map = {}
    p_map = {}
    t_counter = 0
    p_counter = 0

    alignment = alignment_dict["alignment"]
    used_transitions = set()
    # complete replay sync_prod according to alignment
    # assumption: safe net
    curr_marking = get_initial_marking(sync_prod)  # stores the places that contain a token
    for ali_transition in alignment:
        next_transition_set = get_transitions_by_label(ali_transition, sync_prod)
        if len(next_transition_set) == 1:
            next_transition = list(next_transition_set)[0]
            used_transitions.add(next_transition)
        else:
            # retrieve only the one transition that is activated
            next_transition = find_activated_transition(curr_marking, next_transition_set)
            used_transitions.add(next_transition)
        # fire next_transition
        curr_marking = fire_transition(curr_marking, next_transition)

        opt_net.transitions.add(next_transition)
        # connect all preset places with arcs
        __add_pre_set_of_transition(next_transition, opt_net)
        # connect all postset places with arcs
        __add_post_set_of_transition(next_transition, opt_net)

    # remove remaining unused arcs stored in place and transition objects
    __remove_unconnected_and_duplicated_arcs(opt_net)

    # clear all references to unused transitions
    for place in opt_net.places:
        arcs_to_remove = set()
        for arc in place.out_arcs:
            if arc.target not in used_transitions:
                arcs_to_remove.add(arc)
        for arc in arcs_to_remove:
            place.out_arcs.remove(arc)

    # remove all unnecessary 'ending' places (and its arcs) that were not already ending places in sync_prod,
    # i.e. all places that end somewhere in the middle of the synchronous product
    end_places = set()
    for p in sync_prod.places:
        if len(p.out_arcs) == 0:
            end_places.add(p)
    for p in opt_net.places:
        if len(p.out_arcs) == 0 and p not in end_places:
            opt_net = remove_place(opt_net, p)

    return opt_net


def copy_transitions(target_net, source_net):
    for t in source_net.transitions:
        label = t.label
        name = t.name
        new_t = PetriNet.Transition(name, label)
        target_net.transitions.add(new_t)


def __remove_place_from_net(petri_net, place, starting_places=True):
    if starting_places:
        transition: PetriNet.Transition = list(place.out_arcs)[0].target
        arcs_set = transition.in_arcs
    else:
        transition: PetriNet.Transition = list(place.in_arcs)[0].source
        arcs_set = transition.out_arcs
    arc_to_remove: PetriNet.Arc = None
    for arc in arcs_set:
        if starting_places:
            if arc.source == place:
                arc_to_remove = arc
        else:
            if arc.target == place:
                arc_to_remove = arc
    assert arc_to_remove is not None
    if starting_places:
        transition.in_arcs.remove(arc_to_remove)
    else:
        transition.out_arcs.remove(arc_to_remove)

    # remove arc from net
    all_arcs = petri_net.arcs
    for a in all_arcs:
        if a.source == arc_to_remove.source:
            if a.target == arc_to_remove.target:
                petri_net.arcs.remove(a)
                break
    # remove place from net
    all_places = petri_net.places
    place_to_remove = None
    for p in all_places:
        if p.name == place.name:
            place_to_remove = p
            break
    petri_net.places.remove(place_to_remove)


def remove_start_and_end_places(opt_ali_net) -> PetriNet:
    starting_places = set()
    ending_places = set()
    for place in opt_ali_net.places:
        if len(place.in_arcs) == 0:
            starting_places.add(place)
        if len(place.out_arcs) == 0:
            ending_places.add(place)
    for start_place in starting_places:
        __remove_place_from_net(opt_ali_net, start_place, True)
    for end_place in ending_places:
        __remove_place_from_net(opt_ali_net, end_place, False)
    return opt_ali_net


def transform_transition_names(net):
    # convention (>>, A) is a model skip; (A, >>) is a log skip
    all_transitions = net.transitions
    for t in all_transitions:
        old_label = t.label
        if type(old_label) == tuple:
            if old_label[0] == ">>":
                new_label = old_label[1]
                t.label = new_label + "_M"
            elif old_label[1] == ">>":
                new_label = old_label[0]
                t.label = new_label + "_L"
            else:
                new_label = old_label[0]
                t.label = new_label + "_s"


def __find_existing_transition(net, arc_source: PetriNet.Transition):
    all_transitions = net.transitions
    for t in all_transitions:
        if t.name == arc_source.name and t.label == arc_source.label:
            return t


def __arc_already_in_net(net, arc: EventNet.EventArc):
    all_arcs = net.arcs
    for a in all_arcs:
        # go over transition labels+names to mitigate two arcs between the same pair of events
        source_name_equality = (arc.source.name == a.source.name)
        source_label_equality = (arc.source.label == a.source.label)
        target_name_equality = (arc.target.name == a.target.name)
        target_label_equality = (arc.target.label == a.target.label)
        if source_name_equality and target_name_equality and source_label_equality and target_label_equality:
            return True
    return False


def construct_opt_p_ali(opt_ali_net: PetriNet):
    opt_p_alignment = EventNet("Optimal p-Alignment")

    # special case: remove starting and ending places
    opt_ali_net = remove_start_and_end_places(opt_ali_net)

    copy_transitions(opt_p_alignment, opt_ali_net)
    # transform each place between transitions into a dependency (arc), if arc does not already exist
    for place in opt_ali_net.places:
        assert len(pre_set(place)) == 1
        arc_source = list(pre_set(place))[0]
        assert len(post_set(place)) == 1
        arc_target = list(post_set(place))[0]
        temp_arc = EventNet.EventArc(arc_source, arc_target, 1)
        if not __arc_already_in_net(opt_p_alignment, temp_arc):
            original_source = __find_existing_transition(opt_p_alignment, arc_source)
            original_target = __find_existing_transition(opt_p_alignment, arc_target)
            add_arc_from_to(original_source, original_target, opt_p_alignment, type="event_arc")

    transform_transition_names(opt_p_alignment)
    return opt_p_alignment


def apply_from_variant(variant, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a single variant

    Parameters
    -------------
    variant
        Variant (as string delimited by the "variant_delimiter" parameter)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    ------------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}
    trace = variants_util.variant_to_trace(variant, parameters=parameters)

    return apply(trace, petri_net, initial_marking, final_marking, parameters=parameters)


def apply_from_variants_dictionary(var_dictio, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a variants dictionary

    Parameters
    -------------
    var_dictio
        Dictionary of variants (along possibly with their count, or the list of indexes, or the list of involved cases)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}
    dictio_alignments = {}
    for variant in var_dictio:
        dictio_alignments[variant] = apply_from_variant(variant, petri_net, initial_marking, final_marking,
                                                        parameters=parameters)
    return dictio_alignments


def apply_from_variants_list(var_list, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}
    start_time = time.time()
    max_align_time = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME, parameters,
                                                sys.maxsize)
    max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)
    dictio_alignments = {}
    for varitem in var_list:
        this_max_align_time = min(max_align_time_trace, (max_align_time - (time.time() - start_time)) * 0.5)
        variant = varitem[0]
        parameters[Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = this_max_align_time
        dictio_alignments[variant] = apply_from_variant(variant, petri_net, initial_marking, final_marking,
                                                        parameters=parameters)
    return dictio_alignments


def apply_from_variants_list_petri_string(var_list, petri_net_string, parameters=None):
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net_string
        String representing the accepting Petri net

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}

    from pm4py.objects.petri_net.importer.variants import pnml as petri_importer

    petri_net, initial_marking, final_marking = petri_importer.import_petri_from_string(petri_net_string)

    res = apply_from_variants_list(var_list, petri_net, initial_marking, final_marking, parameters=parameters)
    return res


def apply_from_variants_list_petri_string_mprocessing(mp_output, var_list, petri_net_string, parameters=None):
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    mp_output
        Multiprocessing output
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net_string
        String representing the accepting Petri net

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}

    res = apply_from_variants_list_petri_string(var_list, petri_net_string, parameters=parameters)
    mp_output.put(res)


def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, parameters=None):
    """
        Performs the basic alignment search, given a trace net and a net.

        Parameters
        ----------
        trace_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net representing the partial trace
        trace_im: the initial marking of the partial trace
        trace_fm: the final marking of the partial trace
        petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
        initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
        final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
        parameters: :class:`dict` (optional) dictionary containing one of the following:
            Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
            Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            model cost
            Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            synchronous costs
            Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events
            Parameters.PARAM_TRACE_NET_COSTS: :class:`dict` (parameter) mapping between transitions and costs

        Returns
        -------
        dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
        sync_prod: :class: pm4py.objects.petri.net.PetriNet
        """
    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE,
                                                         parameters, False)

    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    trace_net_costs = exec_utils.get_param_value(Parameters.PARAM_TRACE_NET_COSTS, parameters, None)

    # DEBUG
    if True or trace_cost_function is None or model_cost_function is None or sync_cost_function is None:
        sync_prod, sync_initial_marking, sync_final_marking = construct(trace_net, trace_im,
                                                                        trace_fm, petri_net,
                                                                        initial_marking,
                                                                        final_marking,
                                                                        utils.SKIP)
        cost_function = utils.construct_standard_cost_function(sync_prod, utils.SKIP)
    else:
        revised_sync = dict()
        for t_trace in trace_net.transitions:
            for t_model in petri_net.transitions:
                if t_trace.label == t_model.label:
                    revised_sync[(t_trace, t_model)] = sync_cost_function[t_model]

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
            trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, utils.SKIP,
            trace_net_costs, model_cost_function, revised_sync, None)

    max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)

    alignment = apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,
                                utils.SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                max_align_time_trace=max_align_time_trace)

    return_sync_cost = exec_utils.get_param_value(Parameters.RETURN_SYNC_COST_FUNCTION, parameters, False)
    if return_sync_cost:
        # needed for the decomposed alignments (switching them from state_equation_less_memory)
        return alignment, cost_function

    return alignment, sync_prod


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, skip, ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    """
    Performs the basic alignment search on top of the synchronous product net, given a cost function and skip-symbol

    Parameters
    ----------
    sync_prod: :class:`pm4py.objects.petri.net.PetriNet` synchronous product net
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the synchronous product net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the synchronous product net
    cost_function: :class:`dict` cost function mapping transitions to the synchronous product net
    skip: :class:`Any` symbol to use for skips in the alignment

    Returns
    -------
    dictionary : :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states**
    and **traversed_arcs**
    """
    return __search(sync_prod, initial_marking, final_marking, cost_function, skip,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, max_align_time_trace=max_align_time_trace)


def __search(sync_net, ini, fin, cost_function, skip, ret_tuple_as_trans_desc=False,
             max_align_time_trace=sys.maxsize):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    closed = set()

    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]

    use_cvxopt = False
    if lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN or lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP:
        use_cvxopt = True

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

    h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                       ini,
                                                       fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                       use_cvxopt=use_cvxopt)
    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1

    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    while not len(open_set) == 0:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)

        current_marking = curr.m

        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            already_closed = current_marking in closed
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                               incidence_matrix, curr.m,
                                                               fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                               use_cvxopt=use_cvxopt)
            lp_solved += 1

            # 11/10/19: shall not a state for which we compute the exact heuristics be
            # by nature a trusted solution?
            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            # 11/10/2019 (optimization ZA) heappushpop is slightly more efficient than pushing
            # and popping separately
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            if current_marking == fin:
                return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=lp_solved)

        closed.add(current_marking)
        visited += 1

        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            if new_marking in closed:
                continue
            g = curr.g + cost

            queued += 1
            h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = utils.__trust_solution(x)
            new_f = g + h

            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)
