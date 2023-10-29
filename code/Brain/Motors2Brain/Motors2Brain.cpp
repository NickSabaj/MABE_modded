
#include "Motors2Brain.h"

/* MABE calls makeCopy(), makeBrain(), and mutate() automatically.
 * makeCopy() makes a direct copy, and we overrode it because we have special direct-encoded genome.
 * makeBrain() is like a constructor, but with access to the parent, so we can do inheritance.
 * mutate() is called after initial construction so we have a chance to perturbate genome values.
 *
 * The World will call setInput(), update(), and readOutput() to allow the proprioceptive loop.
 * update() allows the agent/brain to perform computation, and read values that were set
 * via setInput(), and place values that will be avialable to readOutput()
 * setInput() places a single continuous value into a single "neuron" / location.
 * getOutput() reads a single continuous value from a single "neuron" / location.
 */

////////////////
// PARAMETERS //
////////////////
std::shared_ptr<ParameterLink<bool>>
Motors2Brain::initializeRandomPL = Parameters::register_parameter("BRAIN_MOTORS2-initializeRandom", false, "Initialize genome randomly, instead of known good initial values");

// Constructor
Motors2Brain::Motors2Brain(int _nrInNodes, int _nrOutNodes, std::shared_ptr<ParametersTable> PT_) : AbstractBrain(_nrInNodes, _nrOutNodes, PT_) {
  development_phase_finished = false; // true if stage2 was run in 1st network update previously
  // columns to be added to ave file
  popFileColumns.clear();
  // configure animat brain based on num ins and num outs
  num_sensory_neurons = _nrInNodes-2; // this brain only works with the motors2world that supplies additional 2 inputs for [hunger, calories]
  num_motor_neurons = _nrOutNodes;
  // we now have enough to create the various neuron count convenience variables...
  num_total_neurons = NUM_EXCITATORY_NEURONS+num_sensory_neurons+NUM_INHIBITORY_NEURONS+num_motor_neurons+NUM_HUNGER_NEURONS;
  num_outer_neurons = NUM_EXCITATORY_NEURONS+num_sensory_neurons;
  allzeros_sensory_neurons = zeros<frowvec>(num_sensory_neurons);
  all_zeros_all_neurons = zeros<frowvec>(num_total_neurons);
  all_zeros_all_neurons_mat = zeros<fmat>(num_total_neurons,num_total_neurons);
  linspace_1_to_num_neurons_by_half = linspace<frowvec>(0.5,num_outer_neurons-0.5,num_outer_neurons);
  theta = repmat(2.0*M_PI * linspace_1_to_num_neurons_by_half / num_outer_neurons, 5, 1).t();
}

auto Motors2Brain::makeBrain( std::unordered_map<std::string, std::shared_ptr<AbstractGenome>> &_genomes) -> std::shared_ptr<AbstractBrain> {
  /* makeBrain is called on a parent when MABE needs to create a new brain.
   * This happens in 2 cases: very first generation (with a bogus parent), 
   * and most commonly when creating new brains in each generation
   */

  // create the new brain that we will return at the end
  std::shared_ptr<Motors2Brain> newBrain = std::make_shared<Motors2Brain>(nrInputValues, nrOutputValues, PT);

  if (Global::update == -1) { // -1 would mean first (progenitor) generation
    // fill genome with known relatively good values
    /* order and range of values are as follows:
     * ex: [a-b]x8 is value in range a through b and there are 8 of them.
     * [0-10]x10
     * [0-1]x2
     * [0-20]x1
     * [0-1]x1
     * [0-2000]x3
     * [0-1]x4
     */
    int c(0);
    // PhiL (ligands)
    newBrain->genome[c++] = 3*M_PI/4;
    newBrain->genome[c++] = M_PI/4;
    newBrain->genome[c++] = 7*M_PI/4;
    newBrain->genome[c++] = 5*M_PI/4;
    newBrain->genome[c++] = 3*M_PI/2;

    // PhiR (receptors)
    newBrain->genome[c++] = 5*M_PI/4;
    newBrain->genome[c++] = 7*M_PI/4;
    newBrain->genome[c++] = 3*M_PI/4;
    newBrain->genome[c++] = M_PI/4;
    newBrain->genome[c++] = M_PI/2;

    newBrain->genome[c++] = 0.1; //sensory_adapt_min_scale = 0.1f; // s0
    newBrain->genome[c++] = 0.5; //sensory_adapt_delta_scale = 0.5f; // a1
    newBrain->genome[c++] = 8.0; //sensory_adapt_delta_threshold = 8.0f; // a2
    newBrain->genome[c++] = 0.5; //homeostatic_target = 0.5f; // a3
    newBrain->genome[c++] = 1'000; //decay_proximity = 1000; // t1
    newBrain->genome[c++] = 1'000; //decay_eating = 1000; // t2
    newBrain->genome[c++] = 1'000; //decay_hunger = 1000; // t3
    newBrain->genome[c++] = 0.3; //decay_proximity_added =  0.3f; // n1
    newBrain->genome[c++] = 0.01; //add_eating_per_calorie = 0.01f; // n2
    newBrain->genome[c++] = 0.01; //add_hunger_per_calorie = 0.01f; // n3
    newBrain->genome[c++] = 0.2; //decay_counterhebb = 0.2f; // n4

    // actually, if user wanted random data, then overwrite
    // all that with something random
    if (initializeRandomPL->get(PT)) {
      c = 0;
      // PhiL (ligands)
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;

      // PhiR (receptors)
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;
      newBrain->genome[c] = Random::getDouble(12.0); ++c;

      newBrain->genome[c] = Random::getDouble(1.0); ++c; //sensory_adapt_min_scale = 0.1f; // s0
      newBrain->genome[c] = Random::getDouble(1.0); ++c; //sensory_adapt_delta_scale = 0.5f; // a1
      newBrain->genome[c] = Random::getDouble(20.0); ++c; //sensory_adapt_delta_threshold = 8.0f; // a2
      newBrain->genome[c] = Random::getDouble(1.0); ++c; //homeostatic_target = 0.5f; // a3
      newBrain->genome[c] = Random::getDouble(500.0,1'500.0); ++c; //decay_proximity = 1000; // t1
      newBrain->genome[c] = Random::getDouble(500.0,1'500.0); ++c; //decay_eating = 1000; // t2
      newBrain->genome[c] = Random::getDouble(500.0,1'500.0); ++c; //decay_hunger = 1000; // t3
      newBrain->genome[c] = Random::getDouble(1.0); ++c; //decay_proximity_added =  0.3f; // n1
      newBrain->genome[c] = Random::getDouble(0.1); ++c; //add_eating_per_calorie = 0.01f; // n2
      newBrain->genome[c] = Random::getDouble(0.1); ++c; //add_hunger_per_calorie = 0.01f; // n3
      newBrain->genome[c] = Random::getDouble(1.0); ++c; //decay_counterhebb = 0.2f; // n4
    }
  } else {
    /* The common case, basic inheritance:
     * if not update -1, then we should inherit
     * the direct-encoded genome values
     * from the parent. (copy from parent
     * to child)
     */
    std::copy(std::begin(genome),std::end(genome),std::begin(newBrain->genome));
  }
  return newBrain;
  // mutate() automatically gets called by MABE after this
}

auto Motors2Brain::serialize(std::string& name) -> DataMap {
  /* This is how the snapshot files with genome
   * data get created.
   * Write out the genome to a string and
   * store it in the returned datamap.
   */
  DataMap data;
  std::stringstream ss;
  
  ss << genome[0];
  for (int i=1; i<genome.size(); i++) { ss << "|" << genome[i]; }

  //data.set(name+"direct_encoded_genome", ss.str());
  data.set(name+"_sites", ss.str());
  data.set(name+"_genomeLength", std::to_string(genome.size()));
  return data;
}

auto Motors2Brain::deserialize(std::shared_ptr<ParametersTable> PT, std::unordered_map<std::string, std::string>& orgData, std::string& name) -> void {
  /* Opposite of serialize(), used when
   * loading organisms from csv files.
   * Parse the genome data and load
   * it into this brain's 'genome'
   * variable
   */
  //auto brainData = orgData[name+"direct_encoded_genome"];
  auto brainData = orgData[name+"_sites"];
  std::vector<double> loaded_values;
  convertCSVListToVector(brainData, loaded_values, '|');
  std::copy( std::begin(loaded_values), std::end(loaded_values), std::begin(genome));
}

auto Motors2Brain::mutate() -> void {
    /* Called during the reproduction cycle after creation.
     * Order and range of values are as follows:
     * ex: [a-b]x8 is value in range a through b and there are 8 of them.
     * [0-10]x10
     * [0-1]x2
     * [0-20]x1
     * [0-1]x1
     * [0-2000]x3
     * [0-1]x4
     */
    int c(0);
    double lower_bound,upper_bound;
    double mu(0.05); // TODO should be a parameter

    // PhiL (ligands)
    lower_bound = 0.0; upper_bound = 12.0;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;

    // PhiR (receptors)
    lower_bound = 0.0; upper_bound = 12.0;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,upper_bound); } ++c;

    lower_bound = 0.0;
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.1); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //sensory_adapt_min_scale = 0.1f; // s0
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //sensory_adapt_delta_scale = 0.5f; // a1
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.5); genome[c]=std::clamp(genome[c],lower_bound,20.0); } ++c; //sensory_adapt_delta_threshold = 8.0f; // a2
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.1); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //homeostatic_target = 0.5f; // a3
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 100); genome[c]=std::clamp(genome[c],lower_bound,2000.0); } ++c; //decay_proximity = 1000; // t1
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 100); genome[c]=std::clamp(genome[c],lower_bound,2000.0); } ++c; //decay_eating = 1000; // t2
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 100); genome[c]=std::clamp(genome[c],lower_bound,2000.0); } ++c; //decay_hunger = 1000; // t3
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.1); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //decay_proximity_added =  0.3f; // n1
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.03); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //add_eating_per_calorie = 0.01f; // n2
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.03); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //add_hunger_per_calorie = 0.01f; // n3
    if (Random::P(mu)) { genome[c] += Random::getNormal(0.0, 0.1 ); genome[c]=std::clamp(genome[c],lower_bound,1.0); } ++c; //decay_counterhebb = 0.2f; // n4
}

auto Motors2Brain::resetBrain() -> void {
  /*
   * This function / method constructs the initial animat and its network.
   * This function at present uses a simple location dependent algorithm
   * to specify the initial animat network. However, it is intended to use
   * a more biological cell-based method later (ask Mark).
   */
  lifetime_t = 0;
  development_phase_finished = false;
  // floats
  sensory_adapt_min_scale = 0; // s0: minimal scale factor in sensory adaptation
  sensory_adapt_delta_scale = 0; // a1: scale for change in sensory adaptation
  sensory_adapt_delta_threshold = 0; // a2: activity threshold for change in sensory adaptation
  homeostatic_target = 0; // a3: homeostatic plasticity constant ("normal" activity levels)
  decay_proximity = 0;// t1: decay constant for proximity related neuromodulator
  decay_eating = 0;   // t2: decay constant for eating related neuromodulator
  decay_hunger = 0;   // t3: decay constant for hunger related neuromodulator
  decay_proximity_added = 0;  // n1: decay of proximity related neuromodulator added
  add_eating_per_calorie = 0;// n2: amount of eating related neuromodulator added per calorie eaten
  add_hunger_per_calorie = 0;// n3: amount of hunger related neuromodulator added per calorie below hunger threshold
  decay_counterhebb = 0; // n4: decay of counterhebb variable
  hebbian_utilization = 0; // (counterHebb) duration of hebbian change
  neuro_mod_proximity = 0; // (neuroModClose) proximity neuromodulator
  neuro_mod_eat = 0; // (neuroModEat) nourishment neuromodulator
  // fmats
  synaptic_weights.zeros(); // S (square matrix)
  neuron_locs.zeros(); // (neuronLocs) %x and y coordinates of each neuron
  // ints
  hebbian_coupling = 1; // (HebbCoupling)
  // reset neural structure
  synaptic_weights = all_zeros_all_neurons_mat; // S (square matrix
  neural_inputs = all_zeros_all_neurons;    // I
  neural_outputs = all_zeros_all_neurons;   // V
  sensory_adapt_sensitivity_scaling_factors = ones<frowvec>(num_sensory_neurons);
  avg_activity = all_zeros_all_neurons;         // v_avg
  avg_hebbian = all_zeros_all_neurons;       // vh_avg
  avg_input = all_zeros_all_neurons;             // I_avg
  avg_avg_hebbian = all_zeros_all_neurons; // meanv_avg
  hebbian_change = zeros<fmat>(42,42);

  firing_rates = all_zeros_all_neurons;
  mean_firing_rates = all_zeros_all_neurons;

  motor_sums = zeros<frowvec>(2); // 2 motors

  //for (auto& kv : genomes) print("genome name:", kv.first);
    /* order and range of values are as follows:
     * ex: [a-b]x8 is value in range a through b and there are 8 of them.
     * [0-10]x10
     * [0-1]x2
     * [0-20]x1
     * [0-1]x1
     * [0-2000]x3
     * [0-1]x4
     */
  int i=0; // read-counter for reading from the genome array (which is 21 sites from genome)
           // note postfix increment (i++) uses i, then increments after evaluation within the brackets.
  frowvec phiL; phiL << genome[i++] << genome[i++] << genome[i++] << genome[i++] << genome[i++] << endr;
  frowvec phiR; phiR << genome[i++] << genome[i++] << genome[i++] << genome[i++] << genome[i++] << endr;
  sensory_adapt_min_scale = genome[i++]; // s0
  sensory_adapt_delta_scale = genome[i++]; // a1
  sensory_adapt_delta_threshold = genome[i++]; // a2
  homeostatic_target = genome[i++]; // a3
  decay_proximity = genome[i++]; // t1
  decay_eating = genome[i++]; // t2
  decay_hunger = genome[i++]; // t3
  decay_proximity_added = genome[i++]; // n1
  add_eating_per_calorie = genome[i++]; // n2
  add_hunger_per_calorie = genome[i++]; // n3
  decay_counterhebb = genome[i++]; // n4

  neuron_locs = zeros<Mat<float>>(num_total_neurons, 2); // (neuronLocs) %x and y coordinates of each neuron
  neuron_locs( span(0,num_outer_neurons-1), 0) += cos((2.0*M_PI * linspace_1_to_num_neurons_by_half.t() / num_outer_neurons) - M_PI_2); // x pos
  neuron_locs( span(0,num_outer_neurons-1), 1) += sin((2.0*M_PI * linspace_1_to_num_neurons_by_half.t() / num_outer_neurons) - M_PI_2); // x pos

  neuron_locs(40,span::all) = frowvec( {-1.2, 0.0} ); // sets m1 location
  neuron_locs(41,span::all) = frowvec( {1.2, 0.0} ); // sets m2 location
  neuron_locs(42,span::all) = frowvec( {0.0, 0.0} ); // sets hunger neuron location

  //%connect network
  const float BLTZ_ALPHA {2.0}, BLTZ_BETA{20.0};
  fmat Rvals(num_total_neurons, 15, fill::zeros); //strength at which neuron expresses each of 5 receptor types based on genes + 10 receptor types reserved for inhibitory neurons
  fmat Lvals(num_total_neurons, 15, fill::zeros); //strength at which neuron expresses each of 5 receptor types based on genes + 10 receptor types reserved for inhibitory neurons
  Mat<float> phiRrep = repmat(phiR, num_outer_neurons, 1);
  Mat<float> phiLrep = repmat(phiL, num_outer_neurons, 1);
  uvec Rmask = find( abs( theta-phiRrep ) < M_PI_4 );
  uvec Lmask = find( abs(theta-phiLrep) < M_PI_4 );
  fmat modR = zeros<fmat>(num_outer_neurons,size(phiR)[1]);
  modR(Rmask) += cos(2*(theta(Rmask)-phiRrep(Rmask)));
  Rvals(span(0,num_outer_neurons-1),span(0,4)) = modR;
  fmat modL = zeros<fmat>(num_outer_neurons,size(phiL)[1]);
  modL(Lmask) += cos(2*(theta(Lmask)-phiLrep(Lmask)));
  Lvals(span(0,num_outer_neurons-1),span(0,4)) = modL;

  Lvals( span(0,num_outer_neurons-1), span(0,1) ).zeros();

  Rvals( span(num_outer_neurons,num_outer_neurons+1), span::all ).zeros();// %= zeros<fmat>(2, 15); // L and R motors, don't send signals
  Rvals( num_outer_neurons+num_motor_neurons, span::all ).zeros();// %= zeros<fmat>(1, 15); // hunger

  Lvals( span(num_outer_neurons,num_outer_neurons+1), span::all ).zeros();// %= zeros<fmat>(2, 15);
  Lvals( num_outer_neurons+0, 0 ) = 1.0; // motor1 (left) (motor only receives signals from neurons with receptor 1)
  Lvals( num_outer_neurons+1, 1 ) = 1.0; // motor2 (right) (motor only receives signals from neurons with receptor 1)
  Lvals( num_outer_neurons+num_motor_neurons, span::all ).zeros();// %= zeros<fmat>(1,15); // hunger
  for (int i=0; i<NUM_INHIBITORY_NEURONS; i++) {
    Lvals(num_outer_neurons+num_motor_neurons+NUM_HUNGER_NEURONS+i, 5+i) = 1.0f;
    Rvals(num_outer_neurons+num_motor_neurons+NUM_HUNGER_NEURONS+i, 10+i) = 1.0f;
    Lvals( span(num_sensory_neurons+((i+0)*4),num_sensory_neurons+(i+1)*4-1), 10+i ).ones();
    Rvals( span(num_sensory_neurons+((i+0)*4),num_sensory_neurons+(i+1)*4-1), 5+i ).ones();
  }

  //%sets S matrix of synapse strength between each neuron using Rvals and Lvals
  const float MAX_SYNAPSE_STRENGTH {30.0}; // maxSynapseStrength
  fmat cross_prods = Lvals * Rvals.t();
  fmat connection_weights = MAX_SYNAPSE_STRENGTH * exp(BLTZ_ALPHA*cross_prods) / (BLTZ_BETA + exp(BLTZ_ALPHA*cross_prods));
  connection_weights( find(connection_weights <= 1.5f) ) *= 0.0f;

  synaptic_weights = connection_weights; // neuron with receptors send signal to neuron with ligands
  synaptic_weights( span::all, span(num_outer_neurons+num_motor_neurons+NUM_HUNGER_NEURONS,num_total_neurons-1) ) *= -1.0f; // inhibitory neurons have negative connections
  synaptic_weights.diag().zeros(); // set diagonals to 0
}

auto Motors2Brain::updateNeuroModulators() -> void {
  // part of the eat() functionality
  neuro_mod_proximity *= (1.0f-(float(FRQ_WORLD_MOVEMENT)/decay_proximity));
  neuro_mod_eat *= (1.0f-(float(FRQ_WORLD_MOVEMENT)/decay_eating));
}

auto Motors2Brain::update() -> void {
  ++lifetime_t; // keeps track of time passing in the brain (inc at fn start ensures start on 1)
  auto& agent = *this;

  // neural refinement ("stage 2")
  if (not agent.development_phase_finished) {
    for (int development_t(1); development_t<=Motors2Brain::NEURAL_GROWTH_PERIOD; development_t++) {
      if ((development_t%Motors2Brain::FRQ_NETWORK_UPDATES)==0)     { agent.spontaneousActivity(); }
      if ((development_t%Motors2Brain::FRQ_HEBBIAN_PLASTICITY)==0)  { agent.hebb({ .reinforce_always=true }); }
    }
    agent.development_phase_finished = true;
  }
  // lifetime ("stage 3")
  // WARNING: start on timestep 1, not 0, because the
  // plasticity functions require a history
  // to operate correctly, and timestep 0
  // would make (if lifetime_t%NUMBER==0) fire on the first step.
  if ((lifetime_t%Motors2Brain::FRQ_NETWORK_UPDATES)==0)        { agent.updateNetwork(); }
  if ((lifetime_t%Motors2Brain::FRQ_WORLD_MOVEMENT)==0)         { agent.updateNeuroModulators(); } // part of eat()
  if ((lifetime_t%Motors2Brain::FRQ_HEBBIAN_PLASTICITY)==0)     { agent.hebb({ .reinforce_always=false }); }
  if ((lifetime_t%Motors2Brain::FRQ_NEURAL_ADAPTATION)==0)      { agent.sensoryAdaptation(); }
  if ((lifetime_t%Motors2Brain::FRQ_HOMEOSTATIC_PLASTICITY)==0) { agent.plasticity(); }
}

auto Motors2Brain::updateNetwork() -> void {
  /*
   * This function updates the network through one time step.
   */

  neural_outputs = clamp(neural_outputs, 1e-5, 1e+5);
  neural_inputs = K_SYNAPSE_DECAY * neural_inputs  +  (synaptic_weights * neural_outputs.t()).t();
  neural_outputs = neural_outputs  -  LEAK * neural_outputs +  neural_inputs; //revises the neural ouput array according to the present values of the neural input and output arrays and the leak variable.
  neural_outputs = arma::max(neural_outputs,all_zeros_all_neurons); // clamp min values to 0 (specify arma::max, not std::max)

      // This is the unique part different from spontaneousActivity
      // vvvvvvvvvvvvvvvvvvvvvvvvvv
  neural_outputs(span(num_outer_neurons,num_outer_neurons+1)) = clamp(neural_outputs(span(num_outer_neurons,num_outer_neurons+1)), 1e-5, 1e+5);
  motor_sums = clamp(motor_sums, 1e-5, 1e+5);
  motor_sums *= MOTOR_DECAY;
  motor_sums += arma::max(neural_outputs(span(num_outer_neurons,num_outer_neurons+1)),zeros<frowvec>(2))*MOTOR_SCALE; // clamp min value to 0 and multiply by MOTOR_SCALE (decay motor input)
  neural_inputs( span(num_outer_neurons,num_outer_neurons+1) ) = motor_sums;
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^
   
  avg_activity = (NEURAL_ALPHA * avg_activity)  +  (NEURAL_BETA * neural_outputs); //running average of activity in all neurons (of recent)
  avg_hebbian = (0.9 * avg_activity)  +  (0.1 * neural_outputs); //running average of activity in all neurons, for Hebb
  avg_input = (0.9 * avg_input)  +  (0.1 * neural_inputs); //running average of input, finer time steps for Hebb
  avg_avg_hebbian = (0.9 * avg_avg_hebbian) + (0.1 * avg_hebbian); //running average of avg_hebbian
  // set motor output
  this->setOutput(0, motor_sums(0));
  this->setOutput(1, motor_sums(1));
}

auto Motors2Brain::description() -> std::string {
  return  "Motors2 Brain\n";
}

auto Motors2Brain::makeCopy(std::shared_ptr<ParametersTable> PT_) -> std::shared_ptr<AbstractBrain> {
  /* makeCopy exists in every module,
   * and it is especially useful if we
   * use any form of elitism, which necessarily
   * calls makeCopy
   */
  if (PT_ == nullptr) {
    PT_ = PT;
  }
  auto newBrain = std::make_shared<Motors2Brain>(nrInputValues, nrOutputValues, PT_);

  for (int i = 0; i < nrOutputValues; i++) {
    newBrain->outputValues[i] = outputValues[i];
  }

  return newBrain;
}

auto Motors2Brain::plasticity() -> void {
  /*
   * Plasticity handles homeostatic plasticity. It prevents a neuron
   * from being active for time. It is supposed to affect neurons
   * receiving input from other neurons. The adaptation function should
   * do similar work for sensory neurons. In this implementation this
   * function seems to be acting on all neurons.
   * The parameter homeostatic_target (a3) should be evolvable.
   */
  // modify the symmetric matrix only in positions (0:41,0:41)
  for (int i{0}; i<num_outer_neurons; ++i) {
    synaptic_weights(i,span(0,41)) *= exp(homeostatic_target-avg_activity(i));
  }
}

auto Motors2Brain::spontaneousActivity() -> void {
  /*
   * spontaneous activity generates correlated activities in neurons
   * with similar properties these are intended to guide early plasticity.
   * Right now I think that this accomplishes little; however it is
   * expected to play a key role in more complex 'brains' (networks).
   */
  int rnum = Random::getInt(4); // makes num from 0 through 4
  neural_outputs( find(SENSOR_TYPES == rnum) ) += 0.001f;
  rnum = num_sensory_neurons + Random::getInt(NUM_EXCITATORY_NEURONS); //saves the pseudorandom result

  /// Original Matlab Code:
  //if ((rnum != num_outer_neurons-1) || (rnum != num_sensory_neurons))
  //  neural_outputs( span(rnum-1,rnum+1) ) += 0.001f;
  //else if (rnum == num_outer_neurons-1)
  //  neural_outputs( span(rnum-1,rnum) ) += 0.001f;
  //else if (rnum == num_sensory_neurons)
  //  neural_outputs( span(rnum,rnum+1) ) += 0.001f;

  /// Reduced Translation of Matlab Code:
  neural_outputs(rnum) += 0.001f;
  if (rnum > num_sensory_neurons) {
    neural_outputs(rnum-1) += 0.001f;
  }
  if (rnum < num_outer_neurons-1) {
    neural_outputs(rnum+1) += 0.001f;
  }

  neural_inputs = K_SYNAPSE_DECAY * neural_inputs  +  (synaptic_weights * neural_outputs.t()).t();
  neural_outputs = neural_outputs  -  LEAK * neural_outputs  +  neural_inputs; //revises the neural ouput array according to the present values of the neural input and output arrays and the leak variable.
  neural_outputs = arma::max(neural_outputs,all_zeros_all_neurons); // clamp min values to 0 (specify arma::max, not std::max)

  avg_activity = (NEURAL_ALPHA * avg_activity)  +  (NEURAL_BETA * neural_outputs); //running average of activity in all neurons
  avg_hebbian = (0.9 * avg_activity)  +  (0.1 * neural_outputs); //running average of activity in all neurons, for Hebb
  avg_input = (0.9 * avg_input)  +  (0.1 * neural_inputs); //running average of input, finer time steps for Hebb
  avg_avg_hebbian = (0.9 * avg_avg_hebbian) + (0.1 * avg_hebbian); //running average of avg_hebbian
}

auto Motors2Brain::hebb(const HebbConfig& cfg) -> void {
  /*
   * Hebb implements the accumulated Hebbian change. Including normalization.
   * The way this is implemented is very much a series of kludges.
   */

  /* reinforce_always is set when in Stage 2 refinement
   * because we want that period to have constant reinforcement
   */
  if ((cfg.reinforce_always) or (neuro_mod_proximity > 1.0f) or (neuro_mod_eat > 1.0f)) { //%if the animat is doing what it should be doing for food and such,
      hebbian_change += hebbian_coupling * exp(-hebbian_utilization) * (avg_hebbian(span(0,num_outer_neurons+num_motor_neurons-1)).t()*avg_hebbian(span(0,num_outer_neurons+num_motor_neurons-1))); //%also make the animat learn! Because it's doing well.
      hebbian_utilization += 1.0; //%but also increase antihebbian activity so plasticity doesn't infinitize
  }


// global S Hebbchange norm
  synaptic_weights.submat(0,0,41,41) += hebbian_change; // sets S equal to whatever S from the other states is times a bit of the Hebb change
  synaptic_weights.submat(0,0,19,39).zeros(); // no connections into sensory neurons
  synaptic_weights(span::all, span(40,41)).zeros(); // no output from motor neurons

  // divisive normalization
  fcolvec summation = sum(synaptic_weights(span::all, span(0,41)),1); // calculate scale factor of each neurons' incoming weights sum from 50
  uvec rowmask = find(summation>50);
  static const uvec colmask = regspace<uvec>(0,41); // makes linear increasing range
  synaptic_weights(rowmask,colmask) %= repmat(50.0f/summation(rowmask),1,42); // divide incoming connections only if connections sum exceeds 50

  synaptic_weights(find(synaptic_weights>10)).fill(10); // individual connections do not exceed 20% of total

  synaptic_weights.diag().zeros(); // set diagonals to 0

  // migrated from Controller2D
  hebbian_utilization = hebbian_utilization*(1.0f-(float(FRQ_HEBBIAN_PLASTICITY)/1'000.0f)*decay_counterhebb); // update the hebbian_utilization matrix with the updated mutation (?) information
  hebbian_utilization = std::max(hebbian_utilization,0.0f); // if any values are negative they should not (and will not be)
}

auto Motors2Brain::sensoryAdaptation() -> void {
  /*
   * This function acts to change the 'gain' or sensitivity of the smell sensors,
   * based on their recent activation history.
   * The parameter a1 should be evolvable.
   */
  for (int i=0; i<num_sensory_neurons; i++) {
    sensory_adapt_sensitivity_scaling_factors(i) = sensory_adapt_sensitivity_scaling_factors(i) + sensory_adapt_delta_scale * (avg_activity(i) - sensory_adapt_delta_threshold);
  }
  sensory_adapt_sensitivity_scaling_factors( find(sensory_adapt_sensitivity_scaling_factors < sensory_adapt_min_scale) ).fill(sensory_adapt_min_scale); //sensitivity capped at min
}

inline
auto Motors2Brain::setInput(const int &inputAddress, const double &value) -> void {
  /* The world calls this function to provide inputs to the brain
   * and the brain can compute with these inputs via its update()
   */
  float newValue = static_cast<float>(value);
  if (inputAddress < num_sensory_neurons) { // normal sensory input
    // scale input
    // sensorInput = sensorScale * sensorInput / s1(i); (original code)
    // we can ignore sensorScale because it is a const (1)
    // so we can reduce that line to the following...
    newValue /= sensory_adapt_sensitivity_scaling_factors(inputAddress);
    neural_inputs(inputAddress) = newValue;
  } else if (inputAddress == num_sensory_neurons+0) { // hunger (within distance 1.0 of prey)
    //neuro_mod_proximity += decay_proximity_added * (animat_hunger * add_hunger_per_calorie);
    neuro_mod_proximity +=   decay_proximity_added * (value         * add_hunger_per_calorie);
  } else if (inputAddress == num_sensory_neurons+1) {
    /* the world passes calories(closest_prey_i) as an input
     * because the brain doesn't know about calories
     */
    neuro_mod_eat += add_eating_per_calorie * value * (readInput(inputAddress-1) * add_hunger_per_calorie);
    // we're gauranteed to have readInput(inputAddress-1) be the correct value
    // because this brain only works with the one world type, and it always provides
    // hunger value if it is also providing calorie value.
  }
  AbstractBrain::setInput(inputAddress, newValue);
}
