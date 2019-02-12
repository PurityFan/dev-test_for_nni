import copy
import os
import json

from hpbandster.core.base_iteration import  Datum

class Run(object):
	"""
		Not a proper class, more a 'struct' to bundle important
		information about a particular run
	"""
	def __init__(self, config_id, budget, loss, info, time_stamps, error_logs):

	def __repr__(self):

	def __getitem__ (self, k):


def extract_HB_learning_curves(runs):

class json_result_logger(object):
	def __init__(self, directory, overwrite=False):

	def new_config(self, config_id, config, config_info):

	def __call__(self, job):

def logged_results_to_HB_result(directory):


class Result(object):
	"""
	Object returned by the HB_master.run function

	This class offers a simple API to access the information from
	a Hyperband run.
	"""
	def __init__ (self, HB_iteration_data, HB_config):

	def __getitem__(self, k):

	def get_incumbent_id(self):

	def get_incumbent_trajectory(self, all_budgets=True, bigger_is_better=True, non_decreasing_budget=True):

	def get_runs_by_id(self, config_id):

	def get_learning_curves(self, lc_extractor=extract_HB_learning_curves, config_ids=None):

	def get_all_runs(self, only_largest_budget=False):

	def get_id2config_mapping(self):

	def _merge_results(self):

	def num_iterations(self):		

	def get_fANOVA_data(self, config_space, budgets=None):




