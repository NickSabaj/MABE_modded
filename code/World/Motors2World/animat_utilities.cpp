#include <animat_utilities>

/* Global variables (data always accessible) */
namespace GLB {

  const static int WORLD_REPS {5};
  const static int NUM_BEST {10};
  const static int NUM_GENERATIONS {400};
  const static int POPULATION_SIZE {100};

  std::vector<std::stringstream> caches;
  std::stringstream popcache;
  std::ofstream log;

  struct LogInitCacheConfig {
    const size_t& population_size;
  };

  auto logInitCaches(const GLB::LogInitCacheConfig& cfg) -> void {
    GLB::caches.resize(cfg.population_size);
    for (auto& cache : GLB::caches) {
      cache.str("");
    }
  }

  auto logWriteCacheToFile(std::stringstream& cache, const std::string& filename) -> void {
    print("writing file:",filename);
    // make sure file is open
    if (not GLB::log.is_open()) GLB::log.open(filename,ios::trunc);
    GLB::log << cache.rdbuf();
    GLB::log.close();
  }
}
