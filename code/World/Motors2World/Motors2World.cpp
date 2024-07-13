
#include "Motors2World.h"
#include <World/Motors2World/animat_utilities>

/* MABE calls the evaluate(population) function for each world,
 * and expects you to store a value in each organism's datamap
 * for "score" so that it can perform selection.
 *
 * To facilitate multithreading at the level of 1 agent's lifetime,
 * all data related to one lifetime of an agent is wrapped into
 * an Experience data structure. This world has a list of Experiences
 * that match up with its list of Organisms, by list position.
 * We reuse an experience data structure and simply reset it
 * when evaluating a new organism.
 */
//
////////////////
// PARAMETERS //
////////////////
std::shared_ptr<ParameterLink<int>> Motors2World::evaluationsPerGenerationPL = Parameters::register_parameter("WORLD_MOTORS2-evaluationsPerGeneration", 5, "Number of times to test each Genome per generation (useful with non-deterministic brains)");

std::shared_ptr<ParameterLink<int>> Motors2World::numThreadsPL = Parameters::register_parameter("WORLD_MOTORS2-numThreads", 0, "Number of threads to use (each member of the population is evaluated on a single thread) 0 implies max physical cores available assuming hyperthreading or equivalent enabled (logical cores / 2).");

std::shared_ptr<ParameterLink<int>> Motors2World::fitnessFunctionPL = Parameters::register_parameter("WORLD_MOTORS2-fitnessFunction", 0, "Fitness function to use for scoring animats (0: food^1.1; 1: (calories/food)^1.1)");

    //dummy parameter for testing mq_conditions.txt functionality
std::shared_ptr<ParameterLink<int>> Motors2World::mq_conditionsDummyVariableBySabajPL = Parameters::register_parameter("WORLD_MOTORS2-mq_conditionsDummyVariableBySabaj", 0, "Dummy variable to test mq_conditions.txt functionality.");



// don't typically need to worry about these 2 parameters,
// as they are for complex simultaneous deme/group simulation
// or odd multi-genome multi-brain hybrids (engineering domain).
std::shared_ptr<ParameterLink<std::string>> Motors2World::groupNamePL = Parameters::register_parameter("WORLD_MOTORS2_NAMES-groupNameSpace", (std::string) "root::", "namespace of group to be evaluated");
std::shared_ptr<ParameterLink<std::string>> Motors2World::brainNamePL = Parameters::register_parameter("WORLD_MOTORS2_NAMES-brainNameSpace", (std::string) "root::", "namespace for parameters used to define brain");

Motors2World::Motors2World(std::shared_ptr<ParametersTable> PT_) : AbstractWorld(PT_) {
  // columns to be added to recorded ave file
  // note how you can add _VAR to automatically calculate variance
  // of something that may have more than 1 value, such as
  // score of an agent that has been evaluated multiple times.
  popFileColumns.clear();
  popFileColumns.push_back("score");
  popFileColumns.push_back("score_VAR"); // specifies to also record the
                                         // variance (performed automatically
                                         // because _VAR)
  popFileColumns.push_back("food_eaten");
  popFileColumns.push_back("food_eaten_VAR");

  // add columns for behavioral output
  popFileColumns.push_back("distance_traveled");
  popFileColumns.push_back("left_turns_15");
  popFileColumns.push_back("left_turns_30");
  popFileColumns.push_back("left_turns_45");
  popFileColumns.push_back("left_turns_60");
  popFileColumns.push_back("right_turns_15");
  popFileColumns.push_back("right_turns_30");
  popFileColumns.push_back("right_turns_45");
  popFileColumns.push_back("right_turns_60");
  popFileColumns.push_back("accel_avg");
  popFileColumns.push_back("moves");

  // generate the smell profiles if not done already
  if (Motors2World::smell_profiles_generated == false) {
    generateSmellProfilesForAllGenerations();
    Motors2World::smell_profiles_generated = true;
  }

  // report thread usage
  std::cout << "Motors2World Threads: " << ((numThreadsPL->get(PT)==0) ? std::thread::hardware_concurrency()/2 : numThreadsPL->get(PT)) << std::endl;
}

/* assigns 1 of 2 profiles to each generation, randomly
 * Okay, so we're only generating probably a subset relative to num generations,
 * so we'll use a getter function to loop reading it, modding by generation number.
 * Generates 1024 different profiles - this relies on the
 * getter function that reads from this as a circular buffer: getSmellProfile(int current_generation)
 */
void Motors2World::generateSmellProfilesForAllGenerations() {
  /* uses the snow arma:: random number generator,
  * but it's okay because we only ever call this once
  * per MABE executable invocation.
  */
  Motors2World::smell_profile_ids_by_generation = randi<Row<int>>(PROFILES_BUFFER_SIZE, distr_param(0,MAX_NUM_PROFILES-1));
}

/* getter function that automatically loops reading from the smell profiles
 * which is okay, because we only need a sufficiently large number of randomly
 * generated profiles.
 */
inline
auto Motors2World::getSmellProfile(int const& generation) -> int {
  return Motors2World::smell_profile_ids_by_generation(generation % PROFILES_BUFFER_SIZE);
}

////////////////
// Experience //
////////////////

auto Motors2World::Experience::reset(const ResetConfig& cfg) -> void {
  linspace_1_to_num_neurons_by_half = linspace<frowvec>(0.5,NUM_INPUTS-0.5,NUM_INPUTS);

  animat_dir = 0.0f;
  rotation_matrix.zeros();

  animat_dir = 0.0f;
  motor_diff = 0.0f;
  brain_output = zeros<frowvec>(Motors2World::NUM_OUTPUTS);
  animat_loc = frowvec{0,0};
  animat_energy = 150.0f;
  animat_hunger = 0.0f;

  food_eaten = 0;
  calories_eaten = 0;
  distance_traveled = 0;

  left_turns_15 = 0;
  left_turns_30 = 0;
  left_turns_45 = 0;
  left_turns_60 = 0;
  right_turns_15 = 0;
  right_turns_30 = 0;
  right_turns_45 = 0;
  right_turns_60 = 0;

  step = 0;
  moves = 0;
  displacement_vec = zeros<frowvec>(Experience::LIFETIME_PERIOD);
  moving_avg = zeros<frowvec>(Experience::LIFETIME_PERIOD - 2);

  // adjust food over evolutionary time
  food_initial_density = 0.0177f;// * pow(0.997f,cfg.generation);
  food_target_density = 0.005f;// * pow(0.997f,cfg.generation);

  prey_locs = zeros<fmat>(NUM_FOOD_ITEMS,3);
  prey_odors = zeros<fmat>(NUM_FOOD_ITEMS,5); // second dim (5) is tied to num sensor types
  // place food randomly
  //   initialize food quantities to 1 at each position
  prey_locs( span::all, 0) += 1.0;
  //   initialize 2D food positions to between -25 and 25 ([0to1]*50-25)
  fmat loc_init_vals = zeros<fmat>(NUM_FOOD_ITEMS,2);
  fill::random(loc_init_vals);
  prey_locs( span::all, span(1,2)) = loc_init_vals * 50 - 25;
  // imprint smell profile on world
  int smell_profile = Motors2World::getSmellProfile(cfg.generation);

  // for fixing the word to a single profile
  //int smell_profile = Motors2World::getSmellProfile(0);
  
  prey_urgency = cfg.generation*0.001f;
  if (cfg.generation > 100){
    prey_urgency = 0.1;
  }

  // for fixing the world to have prey to only kind of ever run away
  //prey_urgency = 0.01;

  // switch up the smell profile every other generation
  // can add more profiles here later
  if (smell_profile == 0) {
    // make high and low calorie foods smell [1,1,1,0,0] and [0,0,0,1,1] respectively.
    prey_odors.submat(cal10mask, uvec {0,1,2}) += 1.0;
    prey_odors.submat(cal30mask, uvec {3,4}) += 1.0;
  } else if (smell_profile == 1) {
    // swap smells
    prey_odors.submat(cal10mask, uvec {3,4}) += 1.0;
    prey_odors.submat(cal30mask, uvec {0,1,2}) += 1.0;
  }
  // radially randomize food
  fcolvec dir_init_vals = zeros<fcolvec>(NUM_FOOD_ITEMS);
  fill::random(dir_init_vals);
  prey_dirs = dir_init_vals * 2 * float{M_PI};

  //initialize neuron locations
  neuron_locs = zeros<Mat<float>>(NUM_INPUTS, 2); // (neuronLocs) %x and y coordinates of each neuron
  // not on below, the original code produced locations for many neurons, not just input neurons
  // but here we don't need that. However, to remain consistent with the kinds of values that were generated
  // in the old code, we must retain the division by a larger number (inputs + excitatory neurons)
  neuron_locs( span::all, 0) += cos((2.0*M_PI * linspace_1_to_num_neurons_by_half.t() / (NUM_INPUTS+20)) - M_PI_2); // x pos
  neuron_locs( span::all, 1) += sin((2.0*M_PI * linspace_1_to_num_neurons_by_half.t() / (NUM_INPUTS+20)) - M_PI_2); // x pos
}

auto Motors2World::Experience::movePrey() -> void {
  float dist {0.0f}, fleex {0.0f}, fleey {0.0f};
  // TODO: these opaque math functions should be refactored
  //       into named functions so the code reads better
  //       and it's more obvious what the math is doing.
  fcolvec dir_init_vals = zeros<fcolvec>(NUM_FOOD_ITEMS);
  fill::random(dir_init_vals);
  prey_dirs += ones<fcolvec>(NUM_FOOD_ITEMS)  * (float{-M_PI} / 6.0f) +
               dir_init_vals                  * (float{ M_PI} / 6.0f) * 2;
  // modify food locations (prey_locs[1 and 2] for x and y) based on prey_dirs
  prey_locs( span::all, 1) += cos(prey_dirs) * 0.1f;
  prey_locs( span::all, 2) += sin(prey_dirs) * 0.1f;
  // if animat is close to prey, then make prey flee
  // (check distance and augment location for each prey)
  for (int preyi=0; preyi<NUM_FOOD_ITEMS; preyi++) {
    // standard euclidean distance: sqrt( (X2-X1)^2 + (Y2-Y1)^2 )
    dist = sqrt(
        pow( prey_locs(preyi,1) - animat_loc(0), 2 )
        +
        pow( prey_locs(preyi,2) - animat_loc(1), 2 )
        );
    if (dist <= 5) {
      fleex = (prey_locs(preyi,1) - animat_loc(0))/dist*(prey_urgency/(dist+0.5f));
      fleey = (prey_locs(preyi,2) - animat_loc(1))/dist*(prey_urgency/(dist+0.5f));
      prey_locs(preyi, 1) += fleex;
      prey_locs(preyi, 2) += fleey;
      prey_dirs(preyi) = atan(((sin(prey_dirs(preyi))*0.1f)+fleey)/((cos((prey_dirs(preyi))*0.1f)+fleex)));
    }
  }
  // limit food positions to artificial boundary
  boundary_length = sqrt(NUM_FOOD_ITEMS/food_target_density)/2;
  // change all x,y locations that are outside the boundary
  // so the positions wrap around. ('toroidal topology')
  prey_dirs( find(prey_locs(span::all,1) >  boundary_length) ) += float{M_PI};
  prey_dirs( find(prey_locs(span::all,2) >  boundary_length) ) += float{M_PI};
  prey_dirs( find(prey_locs(span::all,1) < -boundary_length) ) += float{M_PI};
  prey_dirs( find(prey_locs(span::all,2) < -boundary_length) ) += float{M_PI};
}

auto Motors2World::Experience::getInputSmell(std::shared_ptr<AbstractBrain> brain) -> void {
  /*
   * Smell computes inputs to the smell sensors based on the
   * current position of the prey items and of the animat.
   */
  // set up rotation matrix:
  // [cos(dir), -sin(dir); ([0,2;
  //  sin(dir), cos(dir)]    1,3]
  rotation_matrix.zeros();
  rotation_matrix(uvec{0,3}).fill(cos(animat_dir));
  rotation_matrix(uvec{1,2}).fill(sin(animat_dir));
  rotation_matrix(0,1) *= -1;
  
  // port slow version,
  // then try faster cached SDF version
  // (would yield 50%+ speed boost for this fn)
  // (update: well, this fn is not a bottleneck, so don't bother)
  float dist {0.0f};
  float sensor_accumulation {0.0f};


  // extract only sensor positions from all neuron positions
  // and transpose it in preparation for matrix operations
  // (the absolute location of sensors in the world)
   Mat<float> sensor_locs =       neuron_locs.rows(0,NUM_INPUTS-1               ).t();
   sensor_locs = repmat(animat_loc.t(),1,NUM_INPUTS)                + rotation_matrix*sensor_locs;

   for (int sensori=0; sensori<NUM_INPUTS;                sensori++) {
     sensor_accumulation = 0.0f;
     for (int preyi=0; preyi<NUM_FOOD_ITEMS; preyi++) {
       // standard euclidean distance: sqrt( (X2-X1)^2 + (Y2-Y1)^2 )
       dist = std::sqrt(
                std::pow( prey_locs(preyi,1) - sensor_locs(0,sensori), 2 )
                +
                std::pow( prey_locs(preyi,2) - sensor_locs(1,sensori), 2 )
              );
       sensor_accumulation += prey_locs(preyi,0)* (4/(dist/4+1))*(prey_odors(preyi,sensor_to_odor_mapping[sensori]));
     }
     // set input
     brain->setInput(sensori, sensor_accumulation);
   }
}

auto Motors2World::Experience::getOutputAndMove(std::shared_ptr<AbstractBrain> brain) -> void {
  /* getOutputAndMove() determines motor
   * outputs and update the animat's position.
   */

  // read the brain output into local variable (what used to be motor_sums in matlab)
  brain_output(0) = static_cast<float>(brain->readOutput(0));
  brain_output(1) = static_cast<float>(brain->readOutput(1));

  motor_diff = brain_output(1) - brain_output(0); // brain_output (motor_sums) is reversed in matlab
  motor_diff = clamp(motor_diff, -1.0f, 1.0f);

  // turn by arcsine of half-difference between R and L motor activations
  float turn = asin( motor_diff / 2); // AW
  if (turn < -0.75){ left_turns_60++; } // record big and small turns - AW
  else if (turn < -0.5){ left_turns_45++; }
  else if (turn < -.25){ left_turns_30++; }
  else if (turn < 0) { left_turns_15++; }
  else if (turn < 0.25) { right_turns_15++; }
  else if (turn < 0.5) { right_turns_30++; }
  else if (turn < 0.75) { right_turns_45++; }
  else { right_turns_60++; }

  animat_dir += asin( motor_diff / 2 );
  if (any(brain_output)) {
    // get displacement, but limit steps to 0.5
    float displacement = std::min(sum(brain_output)/2.0f,0.5f);
    distance_traveled += displacement;
    animat_loc += frowvec{float(cos(animat_dir)), float(sin(animat_dir))} * displacement;
    animat_energy -= (ENERGY_RATE_REST * sum(brain_output)) + ENERGY_FROM_MOTION_MULTIPLIER;
    displacement_vec(step) = displacement; // calculate moving average of disp for accel
    if (step > 1 && step < Experience::LIFETIME_PERIOD){
      moving_avg(step-2) = (displacement_vec(step-2)/4) + 
			   (displacement_vec(step-1)/2) + 
			   (displacement_vec(step)/4);
    }
    moves++;
  }
  step++;
}

auto Motors2World::Experience::eat(std::shared_ptr<AbstractBrain> brain) -> void {
  /* This function is supposed to handle all the logistics of eating,
   * including determining whether the animat is close enough to
   * any food to consume it, whether the animat is hungry enough. 
   * It updates an animat's hunger, energy, and the foods available to be eaten.
   * It is also supposed to generate a new food item a moderate distance from the animat;
   * but it seems that actually is placed fairly close
   */

  // euclidean distance computation
  prey_dists = sqrt( square(abs(prey_locs(span::all,1) - animat_loc(0) ))
                     +
                     square(abs(prey_locs(span::all,2) - animat_loc(1) ))
                   );
  animat_hunger = THRESHOLD_HUNGER - animat_energy;

  auto closest_prey_i = prey_dists.index_min();
  auto closest_prey_dist = prey_dists(closest_prey_i);

  // trigger hunger-by-proximity if close to food
  if (closest_prey_dist <= 1.0f) {
    brain->setInput(NUM_INPUTS+0, animat_hunger);
  }
  // if close to food and hungry
  if ((closest_prey_dist <= 0.5f)) {// and (animat_energy <= THRESHOLD_HUNGER)) {
    // eat food
    food_eaten++;
    calories_eaten += calories(closest_prey_i);
    animat_energy += calories(closest_prey_i);
    prey_locs(closest_prey_i,0) -= 1;
    brain->setInput(NUM_INPUTS+1, calories(closest_prey_i));
    // all food now eaten at that location
    if (prey_locs(closest_prey_i,0) <= 0) {
      float bound = sqrt(NUM_FOOD_ITEMS/FOOD_DENSITY_TARGET)/2.0f;
      float dist {0};
      // generate a new food at least 5 units away
      while (dist < 5.0f) {
        prey_locs(closest_prey_i,span::all) = frowvec{ 1.0f , -bound+static_cast<float>(Random::getDouble(1.0))*2*bound , -bound+static_cast<float>(Random::getDouble(1.0))*2*bound };
        dist = sqrt( pow(std::abs(prey_locs(closest_prey_i,1)-animat_loc(0)),2)
                     +
                     pow(std::abs(prey_locs(closest_prey_i,2)-animat_loc(1)),2)
                   );
      }
      prey_dirs(closest_prey_i) = static_cast<float>(Random::getDouble(1.0))*2.0f*float{M_PI};
    }
  }
}
////////////////////
// END Experience //
////////////////////

auto Motors2World::evaluate_single_thread(int analyze, int visualize, int debug) -> void {
  /* keep looping and evaluating 1 organism per loop
   * and finish when there are no more orgs to evaluate
   * Workflow:
   * * get a new org ID from org_ids_to_evaluate queue
   * * if no orgs left, then finish thread and return
   * * otherwise process organism at index ID we got
   * * repeat
   */
  while(true) {
    int org_to_evaluate; // -1=finish thread, otherwise = ID of org to evaluate

    // respecting multithreading data race conditions,
    // get the next organism that has yet to be evaluated
    org_ids_guard.lock();
    if (org_ids_to_evaluate.size() == 0) org_to_evaluate = -1;
    else {
      org_to_evaluate = org_ids_to_evaluate.front();
      org_ids_to_evaluate.pop();
    }
    org_ids_guard.unlock();

    // if no orgs to evaluate found, then finish thread
    if (org_to_evaluate == -1) return;

    // get reference to the organism to evaluate
    std::vector<std::shared_ptr<Organism>>& population = *population_ptr;
    std::shared_ptr<Organism> org = population[org_to_evaluate];

    auto brain = std::dynamic_pointer_cast<Motors2Brain>(org->brains[brainNamePL->get(PT)]);


    int connLength = brain->num_total_neurons; 
    int sliceCount = evaluationsPerGenerationPL->get(PT);
    org->organismLevelConnectomeCube.zeros(connLength, connLength, sliceCount); //Potential error source; unsure if "->" or "." is appropriate. Testing by running, for the sake of quick development.
    org->organismLevelFitnessResults.zeros(sliceCount,1);
    //org->orgSliceCountDebug = sliceCount; 
    org->orgFitSizeDebugFirst = org->organismLevelFitnessResults.n_elem;

    auto& world = experiences[org_to_evaluate]; // convenience and readability
    for (int r = 0; r < evaluationsPerGenerationPL->get(PT); r++) {
      world.reset({.generation=Global::update}); // reset the world

      brain->resetBrain();

      // lifetime ("stage 3")
      // WARNING: start on timestep 1, not 0, because the
      // plasticity functions require a history
      // to operate correctly, and timestep 0
      // would make (if lifetime_t%NUMBER==0) fire on the first step.
      for (int lifetime_t(1); lifetime_t<=Experience::LIFETIME_PERIOD; lifetime_t++) {
        brain->update();
        if ((lifetime_t%Experience::FRQ_WORLD_MOVEMENT)==0) {
          world.movePrey();
          world.getInputSmell(brain);
          world.getOutputAndMove(brain);
          world.eat(brain);
          if (visualize) { writeVisualizationData(experiences[org_to_evaluate]); }
        }
      } // end lifetime ("stage 3")
      if (visualize == 1) /*only save 1 then quit*/ { exit(0); }
      float fitness;
      if (not world.food_eaten) {
        fitness = 0;
      } else {
        int fitnessFunction = fitnessFunctionPL->get(PT);
        switch (fitnessFunction) {
          case 0:
            fitness = pow(world.food_eaten,1.1f); // (simpler fitness function, just food eaten)
            break;
          case 1:
            fitness = pow(world.calories_eaten/world.food_eaten,1.1f); // average calories (to do well, animats must learn to detect high calorie foods)
            break;
          case 2:
            fitness = pow(world.calories_eaten,1.1f); // average calories (to do well, animats must learn to detect high calorie foods)
            break;
          case 3:
            fitness = pow(pow(world.calories_eaten,2.0f)/world.food_eaten,1.1f); // average calories (to do well, animats must learn to detect high calorie foods)
            break;

        }
      }
      // record the score (and any other properties you wish)
      // ('append' for list vs 'set' for single entry numbers, strings, etc.)
      // 'score' is used for evolutionary selection by MABE automatically
      org->dataMap.append("score", fitness);
      org->dataMap.append("food_eaten", world.food_eaten);
      org->dataMap.append("distance_traveled", world.distance_traveled);
      org->dataMap.append("left_turns_15", world.left_turns_15);
      org->dataMap.append("left_turns_30", world.left_turns_30);
      org->dataMap.append("left_turns_45", world.left_turns_45);
      org->dataMap.append("left_turns_60", world.left_turns_60);
      org->dataMap.append("right_turns_15", world.right_turns_15);
      org->dataMap.append("right_turns_30", world.right_turns_30);
      org->dataMap.append("right_turns_45", world.right_turns_45);
      org->dataMap.append("right_turns_60", world.right_turns_60);
      org->dataMap.append("accel_avg", mean(world.moving_avg));
      org->dataMap.append("moves", world.moves);
      org->organismLevelConnectomeCube.slice(r) = brain->synaptic_weights;
      org->organismLevelFitnessResults(r) = fitness;
      org->orgFitSizeDebugSecond = org->organismLevelFitnessResults.n_elem;
    }
  }
}

void Motors2World::evaluate(std::map<std::string, std::shared_ptr<Group>> &groups, int analyze, int visualize, int debug) {
  // evaluate() is a MABE API function called automatically for each generation
  // We invoke an 'evaluate_single_thread(org)' for each organism
  // allowing parallel computation.
  // All data local to a single organism is contained in an Experience class instance
  // such as position, number of foods, current calorie profile, num food eaten, etc.
  // The visualize and debug flags are up for you to decide
  // what they mean, and you can set in settings.

  int popSize = groups[groupNamePL->get(PT)]->population.size();
  // save population to World-wide pointer so all threads have access to it
  population_ptr = &groups[groupNamePL->get(PT)]->population;
  // populate list of ids with linear increasing list of indices
  // each thread will claim and remove one id and evaluate that organism
  for (int i=0; i<popSize; i++) org_ids_to_evaluate.push(i);
  // create pool of Experience instances, 1 for each organism
  // but we can reuse the pool if it already exists.
  experiences.resize(popSize); // does nothing if already requested size
  // create and start the thread pool...
  // if no threads set, then set to max cores found
  int num_threads = numThreadsPL->get(PT);
  if (num_threads == 0) num_threads = std::thread::hardware_concurrency()/2;
  // create/start each thread to run evaluate_single_thread()
  for (int i=0; i<num_threads; i++) threads.push_back(std::thread(&Motors2World::evaluate_single_thread, this, analyze, visualize, debug));
  // wait for all threads to finish
  for (auto& thread:threads) thread.join();
  // we can't reuse threads in a good way, so just deconstruct them
  threads.clear();
}

std::unordered_map<std::string, std::unordered_set<std::string>>
Motors2World::requiredGroups() {

  ////////////////
  //Dummy variable section BEGIN; -Sabaj, 5/21/2022
  ////////////////
      //dummy variable to test mq_conditions.txt functionality
       int mq_conditionsDummyVariableBySabaj= mq_conditionsDummyVariableBySabajPL->get(PT);
  ////////////////
  //Dummy variable section END; -Sabaj, 5/21/2022
  ////////////////



  // MABE API function automatically called to get the required information
  // for how to configure the organism (brain) for this task
  // usually, this is only the number of inputs and number of outputs
  // this task requires.
  // There is also the ability to specify how many individual brains (suborganisms)
  // this task requires, and what their in-out configuration should be.
  // We could also specify what demes/groups should exist and their requirements.
  // See the wiki for more information.
  return {{groupNamePL->get(PT),
        {"B:" + brainNamePL->get(PT) + "," + std::to_string(/*inputs*/Motors2World::NUM_INPUTS+2) + "," + std::to_string(/*outputs*/Motors2World::NUM_OUTPUTS)}}};
        // NUM_INPUTS+2 because 2 extra inputs: hunger, calories-just-eaten
  // requires a root group and a brain (in root namespace) and no addtional
  // genome,
}

auto Motors2World::writeVisualizationData(Experience& experience) -> void {
  if (numThreadsPL->get(PT) != 1) {
    std::cout << "Error Motors2World::writeVisualizationData(): Please only visualize data using one thread. Set parameter WORLD_MOTORS2-numThreads to 1." << std::endl;
    exit(1);
  }
  experience.writeVisualizationData();
}

auto Motors2World::Experience::writeVisualizationData() -> void {
  // file format is:
  // positional
  // <x>,<y>,<dir>
  // <numfoods>
  // <food1quantity>,<food1x>,<food1y>
  // <food2quantity>,<food2x>,<food2y>
  // ...
  // Then, the file is read using the vis:
  // ./vis --2d --script=readcsv.gd --file=animat_data.csv
  const std::string visfilename {"animat_data.csv"};
  FileManager::writeToFile(visfilename, "positional");
  std::vector<double> pred_vec = conv_to<std::vector<double>>::from(animat_loc); // x,y
  // write <x>,<y>,<dir>
  FileManager::writeToFile(visfilename, std::to_string(pred_vec[0])+","+std::to_string(pred_vec[1])+","+std::to_string(animat_dir));
  // write <numfoods>
  FileManager::writeToFile(visfilename, std::to_string(prey_locs.n_rows));
  // write <food1quantity>,<food1x>,<food1y> ...
  for (int i=0; i<prey_locs.n_rows; i++) {
    FileManager::writeToFile(visfilename, std::to_string(prey_locs(i,0))+","+std::to_string(prey_locs(i,1))+","+std::to_string(prey_locs(i,2)));
  }
}
