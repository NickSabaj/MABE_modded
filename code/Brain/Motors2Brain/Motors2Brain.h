
#pragma once

#include <cmath>
#include <memory>
#include <set>

#include <thread> // for multithreading
#include <algorithm> // various useful algorithm functions
#include <climits> // INT_MAX to make random nums into floats
#include <vector> // c++'s standard dynamic array
#include <string> // string (text) support
#include <iostream> // terminal & file I/O
#include <cmath> // constants, and math fns, M_PI, etc.
#include <random> // distributions and RNGs
#include <experimental/iterator> // NOTE: 'make_ostream_joiner' will likely become 'ostream_joiner' in c++20

#include <Genome/AbstractGenome.h>
#include <Utilities/Random.h>
#include <Brain/AbstractBrain.h>
#include <World/Motors2World/animat_utilities>

// TODO: make setInput & setOutput API-compliant

class Motors2Brain : public AbstractBrain {
public: // explicitly make everything available (MABE design philosophy)

  ////////////////
  // PARAMETERS //
  ////////////////
  static std::shared_ptr<ParameterLink<bool>> initializeRandomPL;

  Motors2Brain() = delete;
  Motors2Brain(int /*_nrInNodes*/, int /*_nrOutNodes*/, std::shared_ptr<ParametersTable> /*PT_*/ = nullptr);

  ~Motors2Brain() = default;

  auto serialize(std::string & /*name*/) -> DataMap override;
  auto deserialize(std::shared_ptr<ParametersTable> PT, std::unordered_map<std::string, std::string> & /*orgData*/, std::string & /*name*/) -> void override;

  auto updateNetwork() -> void;
  auto update() -> void override;

  auto makeBrain(std::unordered_map<std::string, std::shared_ptr<AbstractGenome>> &/*_genomes*/) -> std::shared_ptr<AbstractBrain> override;
  auto mutate() -> void override;

  auto requiredGenomes() -> std::unordered_set<std::string> override {
    /* this brain uses no MABE genomes, it is direct-encoded */
    return {};
  }
  auto getStats(std::string & prefix) -> DataMap override { return (DataMap{}); }

  std::string description() override;
  std::string getType() override { return "Motors2"; }

  auto resetBrain() -> void override;

  
  auto makeCopy(std::shared_ptr<ParametersTable> /*PT_*/ = nullptr) -> std::shared_ptr<AbstractBrain> override;

  inline
  auto setInput(const int & /*inputAddress*/, const double & /*value*/) -> void override;

  // ***************************  
  std::array<double,21> genome;
  bool development_phase_finished;
  int lifetime_t {0}; // counter of time ticks (num times update() was called) to determine when neural functions should run
  float motor_diff; // same # as motor neurons in agent
  float sensory_adapt_min_scale, // s0: minimal scale factor in sensory adaptation
	sensory_adapt_delta_scale, // a1: scale for change in sensory adaptation
	sensory_adapt_delta_threshold, // a2: activity threshold for change in sensory adaptation
	homeostatic_target, // a3: homeostatic plasticity constant ("normal" activity levels)
	decay_proximity,// t1: decay constant for proximity related neuromodulator
	decay_eating,   // t2: decay constant for eating related neuromodulator
	decay_hunger,   // t3: decay constant for hunger related neuromodulator
	decay_proximity_added,  // n1: decay of proximity related neuromodulator added
	add_eating_per_calorie,// n2: amount of eating related neuromodulator added per calorie eaten
	add_hunger_per_calorie,// n3: amount of hunger related neuromodulator added per calorie below hunger threshold
	decay_counterhebb; // n4: decay of counterhebb variable
  fmat hebbian_change; // (Hebbchange)
  int hebbian_coupling {1}; // (HebbCoupling)
  float hebbian_utilization {0}; // (counterHebb) duration of hebbian change
  Mat<float> firing_rates,
	     mean_firing_rates;
  int num_sensory_neurons, // Ns (for instance, 20. set by world)
	    num_motor_neurons, // Nm (for instance, 2. set by world)
	    num_total_neurons, // (for instance, 48 [NUM_EXCITATORY_NEURONS + num_sensory_neurons + NUM_INHIBITORY_NEURONS + num_motor_neurons + NUM_HUNGER_NEURONS]
	    num_outer_neurons; // Nr = Ns+Ne (for instance, 40 [NUM_EXCITATORY_NEURONS + num_sensory_neurons]
  static const inline int 
	    NUM_EXCITATORY_NEURONS {20}, // Ne
	    NUM_HUNGER_NEURONS {1}, // Nh
	    NUM_INHIBITORY_NEURONS {5}, // Ni
	    ANIMAT_RADIUS {1};
  static const inline float K_SYNAPSE_DECAY {0.7}, // decay multiplier per time step
	      MOTOR_DECAY {0.9}, // decay multiplier per time step
	      HEBB_COUPLING {1.0},
	      LEAK {0.1}, // Leak
	      INPUT_SCALE {0.0001}, // scaling input to excitatory interneurons
	      SENSOR_SCALE {1.0}, // scaling input to sensor neurons
	      MOTOR_SCALE {0.001}; // scaling input to motor neurons
  static const inline float NEURAL_ALPHA {float(5'999)/float(6'000)}, //constants used for modifying neural input and output activities. (5999/6000)
			    NEURAL_BETA = {float(1)/float(6'000)};
  float neuro_mod_proximity {0}, // (neuroModClose) proximity neuromodulator
	neuro_mod_eat {0}; // (neuroModEat) nourishment neuromodulator
  //Mat<int> sensor_types(NUM_TOTAL_NEURONS, 2, fill::zeros);
  inline static const Mat<int> SENSOR_TYPES {0,1,2,3,4,4,3,2,1,0,0,1,2,3,4,4,3,2,1,0};

  fmat synaptic_weights; // S (square matrix)
  Mat<float> neuron_locs; // (neuronLocs) %x and y coordinates of each neuron 
  frowvec neural_inputs, // I
	  neural_outputs, // V
	  avg_activity, // v_avg (average activity used for hebb function)
	  avg_hebbian, // vh_avg
	  avg_input, // I_avg
	  avg_avg_hebbian, // avg of vh_avg
	  motor_sums, // [m1InputSum,m2InputSum]
	  sensory_adapt_sensitivity_scaling_factors; // s1
  frowvec all_zeros_all_neurons;
  fmat all_zeros_all_neurons_mat;
  frowvec allzeros_sensory_neurons;
  frowvec linspace_1_to_num_neurons_by_half;
  fmat theta;
	static const int NEURAL_GROWTH_PERIOD {300'000}; // (60,000)
		   //FRQ_WORLD_MOVEMENT {100}; // fB: frequency of world/movement updates (milliseconds)
  static const inline int FRQ_NETWORK_UPDATES {10}, // fA: frequency of network updates(every 10 milliseconds)
			                  FRQ_NEURAL_ADAPTATION {100}, // 5'000 // fC: frequency of neural adaptation (every 5 seconds)
			             FRQ_HOMEOSTATIC_PLASTICITY {300'000}, // 60'000 //fD: frequency of homeostatic plasticity (every minute)
			                 FRQ_HEBBIAN_PLASTICITY {100}, // fH: frequency of hebbian plasticity (every 100 milliseconds)
                           FRQ_WORLD_MOVEMENT {100};
  // ***************************  
  auto updateNeuroModulators() -> void;
  auto plasticity() -> void;
  auto spontaneousActivity() -> void;
  struct HebbConfig {
    bool reinforce_always {false};
  };
  auto hebb(const HebbConfig& /*cfg*/) -> void;
  auto sensoryAdaptation() -> void;
};

inline 
auto Motors2Brain_brainFactory(int ins, int outs, std::shared_ptr<ParametersTable> PT) -> std::shared_ptr<AbstractBrain> {
  return std::make_shared<Motors2Brain>(ins, outs, PT);
}

