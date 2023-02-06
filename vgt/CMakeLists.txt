set(SOURCES utils.cpp utils.h RandomEngine.cpp RandomEngine.h
        IndexSet.cpp IndexSet.h KDTree.cpp KDTree.h)

set(VG_SOURCES algo/VoronoiGraph.cpp algo/VoronoiGraph.h algo/kernels.cpp algo/kernels.h ${SOURCES})

convertIntoHeader(cl/voronoi.cl cl/voronoi_cl.h voronoi_kernel_sources)

link_libraries(cnpy)

add_library(vgt MODULE python_bindings.cpp ${VG_SOURCES}
        algo/cell_kernels.h algo/cell_kernels.cpp
        algo/kernels_gpu.h algo/kernels_gpu.cpp
        algo/VoronoiDensityEstimator.h algo/VoronoiDensityEstimator.cpp
        extra/GabrielGraph.cpp extra/GabrielGraph.h
        extra/KDE.cpp extra/KDE.h
        cl/voronoi.cl cl/voronoi_cl.h)
target_link_libraries(vgt libclew libgpu  ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(vgt PRIVATE ${PYTHON_INCLUDE_DIRS})
