
HB = BOHB(	configspace = config_space,
				run_id = args.run_id,
                eta=3,min_budget=27, max_budget=243,
                host=ns_host,
				nameserver=ns_host,
				nameserver_port = ns_port,
				ping_interval=3600,	
		)

# Instance CG_BOHB
cg.top_n_percent = 15
cg.configspace = CS.UniformFloatHyperparameter('x', lower=0, upper=1)
cg.bw_factor = 3
cg.min_bandwith = 1e-3
cg.min_points_in_model = None -> 2
cg.num_samples = 64
cg.random_fraction = 1/3
hps = [x]
# cg.kde_vartypes = ""
# cg.vartypes = []
cg.kde_vartypes = "c"
cg.vartypes = [0]
cg.cat_probs = []
cg.configs = dict()
cg.losses = dict()
cg.good_config_rankings = dict()
cg.kde_models = dict()

# Instance BOHB
# master __init__
self.working_directory = '.'
self.logger = logging.getLogger('hpbandster')
self.result_logger = None
self.config_generator = cg # Instance of CG_BOHB
self.time_ref = None
self.iterations = []
self.jobs = []
self.num_running_jobs = 0
self.job_queue_sizes = (-1, 0)
self.user_job_queue_sizes = (-1, 0)
self.dynamic_queue_size = True
self.warmstart_iteration = []
self.thread_cond = threading.Condition()
'''
self.config = {
	'time_ref'   : None
}
'''
self.dispatcher = Dispatcher( self.job_callback, queue_callback=self.adjust_queue_size, run_id=run_id, ping_interval=ping_interval, nameserver=nameserver, nameserver_port=nameserver_port, host=host)
self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
self.dispatcher_thread.start()

# BOHB __init__
self.eta = 3
self.min_budget = 27
self.max_budget = 243
# precompute some HB stuff
self.max_SH_iter = 3
# -int(np.log(min_budget/max_budget)/np.log(eta)) + 1 = 2 + 1
self.budgets = array[(27, 81, 243)]
# max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))
self.config = {
    'time_ref'   : None
	'eta'        : eta,
	'min_budget' : min_budget,
	'max_budget' : max_budget,
	'budgets'    : self.budgets,
	'max_SH_iter': self.max_SH_iter,
	'min_points_in_model' : min_points_in_model,
	'top_n_percent' : top_n_percent,
	'num_samples' : num_samples,
	'random_fraction' : random_fraction,
	'bandwidth_factor' : bandwidth_factor,
	'min_bandwidth': min_bandwidth
}




res = HB.run(	n_iterations = 4,
					min_n_workers = 4		# BOHB can wait until a minimum number of workers is online before starting
		)

iteration_kwargs = {
    'result_logger': None
}
self.time_ref = time.time()
self.config['time_ref'] = time.time()

while True:
    next_run = None
    for i in self.active_iterations(): # 在self.iteration=[]中选择还没完成的
        next_run = self.iterations[i].get_next_run()
		if not next_run is None: break
    
    if not next_run is None:
		self._submit_job(*next_run)
		continue
	else:
	    if n_iterations > 0:	#we might be able to start the next iteration
			self.iterations.append(self.get_next_iteration(len(self.iterations), iteration_kwargs))
			n_iterations -= 1
			continue

	# at this point there is no imediate run that can be scheduled,
	# so wait for some job to finish if there are active iterations
	if self.active_iterations():
		self.thread_cond.wait()
	else:
		break
	
    self.thread_cond.release()
	
for i in self.warmstart_iteration:
	i.fix_timestamps(self.time_ref)

ws_data = [i.data for i in self.warmstart_iteration]
		
return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.config)




SuccessiveHalving(BaseIteration):

self.data = {}
self.is_finished = False
self.HPB_iter = 4
self.stage = 0					# internal iteration, but different name for clarity
self.budgets = self.budgets[(-s-1):]
self.num_configs = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
self.actual_num_configs = [0]*len(num_configs)
self.config_sampler = self.config_generator.get_config
self.num_running = 0
self.logger=logging.getLogger('hpbandster')
self.result_logger = None

CG_BOHB.get_config()

self.logger.debug('start sampling a new configuration.')

# 如果是第一(iteration)轮sample

	sample = None
	info_dict = {}
	sample =  self.configspace.sample_configuration() # 随机sample # x = 0.4
	info_dict['model_based_pick'] = False
	best = np.inf
	best_vector = None
	sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
				configuration_space=self.configspace,
				configuration=sample.get_dictionary()
			).get_dictionary()
	return sample, info_dict

# 如果是之后的轮次
