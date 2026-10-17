#include "bindings.hpp"

NB_MODULE(_core, module) {
    module.doc() =
        "Native Catan rules engine, flat action space, value expert, observations, "
        "batched environments and MCTS.";
    cppanatron::python::bind_build_info(module);
    cppanatron::python::bind_types(module);
    cppanatron::python::bind_maps(module);
    cppanatron::python::bind_game(module);
    cppanatron::python::bind_codecs(module);
    cppanatron::python::bind_graph(module);
    // The tracker before the VecEnv and the SearchPool: their defaults name TrackerMode.
    cppanatron::python::bind_tracker(module);
    cppanatron::python::bind_vec_env(module);
    cppanatron::python::bind_observation(module);
    cppanatron::python::bind_search(module);
    cppanatron::python::bind_search_pool(module);
}
