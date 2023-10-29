
#pragma once

#include <World/AbstractWorld.h>

#include <cstdlib>
#include <thread> // for multithreading
#include <algorithm> // various useful algorithm functions
#include <climits> // INT_MAX to make random nums into floats
#include <vector> // c++'s standard dynamic array
#include <string> // string (text) support
#include <iostream> // terminal & file I/O
#include <cmath> // constants, and math fns, M_PI, etc.
#include <random> // distributions and RNGs
#include <experimental/iterator> // NOTE: 'make_ostream_joiner' will likely become 'ostream_joiner' in c++20

// import references to the brain (is this still needed? TODO)
#include <Brain/Motors2Brain/Motors2Brain.h>

// import references from the utilities (prune unnecessary from utilities TODO)
#include <World/Motors2World/animat_utilities>


// *************** beginning of world class definition ***************

class Motors2World : public AbstractWorld {
public:
  // if number of inputs and outputs were dynamic
  //static std::shared_ptr<ParameterLink<int>> numberOfInputsPL;
  //static std::shared_ptr<ParameterLink<int>> numberOfOutputsPL;
  static std::shared_ptr<ParameterLink<int>> evaluationsPerGenerationPL;
  static std::shared_ptr<ParameterLink<int>> numThreadsPL;
  static std::shared_ptr<ParameterLink<int>> fitnessFunctionPL; // index of which fitness function to use

  //dummy variable for testing mq_conditions.txt functionality
  static std::shared_ptr<ParameterLink<int>> mq_conditionsDummyVariableBySabajPL;
  //int mq_conditionsDummyVariableBySabaj;


  // int mode;
  // int numberOfOutputs;
  // int evaluationsPerGeneration;

  static std::shared_ptr<ParameterLink<std::string>> groupNamePL;
  static std::shared_ptr<ParameterLink<std::string>> brainNamePL;

  Motors2World(std::shared_ptr<ParametersTable> PT_ = nullptr);
  virtual ~Motors2World() = default;

  std::vector<std::shared_ptr<Organism>>* population_ptr;
  std::queue<int> org_ids_to_evaluate; // indexes into population, remove id when evaluated
  std::mutex org_ids_guard; // locks org_ids_to_evaluate
  std::mutex data_guard;
  std::vector<std::thread> threads;

  auto evaluate_single_thread(int analyze, int visualize, int debug) -> void;
  void evaluate(std::map<std::string, std::shared_ptr<Group>> &groups, int analyze, int visualize, int debug);

  virtual std::unordered_map<std::string, std::unordered_set<std::string>> requiredGroups() override;


  class Experience {
    // Represents one lifetime experience
    // including all pred and prey positions,
    // hunger levels, etc.
    // These are separated into a class
    // to facilitate parallel computation of agents
    // so their lifetime experiences don't confound
    // each other (variables, function calls, etc.).
    public:
      static const int LIFETIME_PERIOD {3'000'000}; // lifetime: amount of simulated millisecond timesteps (600,000)
      static const inline int NUM_FOOD_ITEMS {50};
      static const inline float ENERGY_RATE_REST {0.01}, // rate of burning energy at rest
                   ENERGY_FROM_MOTION_MULTIPLIER {0.01};
      static const inline float THRESHOLD_HUNGER {150.0}; // (hungerThreshold) above equals hunger
      static const inline float FOOD_DENSITY_INITIAL {0.0177}, // initialDensity: default initial density of food
                   FOOD_DENSITY_TARGET {0.005}; // foodDensity: default normal density of food
      static const int FRQ_WORLD_MOVEMENT {100}; // fB: frequency of world/movement updates (milliseconds)
      static const inline std::array sensor_to_odor_mapping {0,1,2,3,4,4,3,2,1,0,0,1,2,3,4,4,3,2,1,0}; // for nicely distributing odors among sensor (if it makes a difference...)
      static const inline frowvec calories {resize(repmat(frowvec{10,10,30},1,ceil(NUM_FOOD_ITEMS/3.0)),1,NUM_FOOD_ITEMS)};
      static const inline uvec cal10mask {find(calories == 10)};
      static const inline uvec cal30mask {find(calories == 30)};
      float animat_energy; // (energy)
      float animat_hunger; // (hunger)
      frowvec linspace_1_to_num_neurons_by_half;
      Mat<float> neuron_locs; // (neuronLocs) %x and y coordinates of each neuron 
      fmat prey_locs, // (foodLocs) NUM_FOOD_ITEMS x 3 (quantity at pos, xpos, ypos)
           prey_odors, // (foodOdors) NUM_FOOD_ITEMS x 5 (initialized once by generateSmellProfilesForAllGenerations(generations))
           prey_dists; // (FoodDist or dist) euclidean distances from animat to all prey
      frowvec brain_input; // localized copy of readOutput results
      frowvec brain_output; // localized copy of readOutput results
      fcolvec prey_dirs; // (foodDir) food items have agency, and thus direction or heading for their movement
      frowvec animat_loc; // animat x,y position
      float animat_dir; // animat heading
      float prey_urgency;
      Mat<float> rotation_matrix { {0.0f, 0.0f}, {0.0f, 0.0f} };
      // row vec of [10,10,30] repeated (NUM_FOOD_ITEMS/3) times
      // make long row vector of calorie values, repeats N=1/3 num foods.
      float  motor_diff;

      int food_eaten {0};
      float calories_eaten {0};
      float distance_traveled {0};
	
      int left_turns_15 {0};
      int left_turns_30 {0};
      int left_turns_45 {0};
      int left_turns_60 {0};
      int right_turns_15 {0};
      int right_turns_30 {0};
      int right_turns_45 {0};
      int right_turns_60 {0};
  
      int step {0};
      int moves {0};
      frowvec displacement_vec;
      frowvec moving_avg;

      // externally-controlled values (annealing)
      float food_initial_density, // (foodInitialDensity) start-of-world food density
            food_target_density; // (foodDensity) stable world food density
      // preyMove
      float boundary_length;
      struct ResetConfig {
        const int& generation;
      };
      auto reset(const ResetConfig& /*config*/) -> void;
      auto movePrey() -> void;
      auto getInputSmell(std::shared_ptr<AbstractBrain> /*agent*/) -> void;
      auto getOutputAndMove(std::shared_ptr<AbstractBrain> /*agent*/) -> void;
      auto eat(std::shared_ptr<AbstractBrain> /*agent*/) -> void;
      auto writeVisualizationData() -> void;

  };
  std::vector<Experience> experiences;

  static const int NUM_INPUTS {20};
  static const int NUM_OUTPUTS {2};
  static const inline int MAX_NUM_PROFILES {2};
  static const inline int PROFILES_BUFFER_SIZE {1024}; // used to make 'enough' profiles we then loop through after this x generations

  static inline Row<int> smell_profile_ids_by_generation; // generate 1024 profiles, and loop it (mod by generation)
  static inline bool smell_profiles_generated {false};

  void generateSmellProfilesForAllGenerations();
  static int getSmellProfile(int const& /*generation*/);
  void writeVisualizationData(Experience& /*experience*/); // calls experience.writeVisualizationData after checking numThreads==1
};

