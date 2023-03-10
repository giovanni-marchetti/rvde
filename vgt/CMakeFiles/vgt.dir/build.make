# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/giovanni/anaconda3/envs/vgtenv/bin/cmake

# The command to remove a file.
RM = /home/giovanni/anaconda3/envs/vgtenv/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/giovanni/Downloads/supplementary/rvde-code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giovanni/Downloads/supplementary/rvde-code

# Include any dependencies generated for this target.
include vgt/CMakeFiles/vgt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include vgt/CMakeFiles/vgt.dir/compiler_depend.make

# Include the progress variables for this target.
include vgt/CMakeFiles/vgt.dir/progress.make

# Include the compile flags for this target's objects.
include vgt/CMakeFiles/vgt.dir/flags.make

vgt/cl/voronoi_cl.h: vgt/cl/voronoi.cl
vgt/cl/voronoi_cl.h: libs/gpu/hexdumparray
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating cl/voronoi_cl.h"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && ../libs/gpu/hexdumparray /home/giovanni/Downloads/supplementary/rvde-code/vgt/cl/voronoi.cl /home/giovanni/Downloads/supplementary/rvde-code/vgt/cl/voronoi_cl.h voronoi_kernel_sources

vgt/CMakeFiles/vgt.dir/python_bindings.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/python_bindings.cpp.o: vgt/python_bindings.cpp
vgt/CMakeFiles/vgt.dir/python_bindings.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object vgt/CMakeFiles/vgt.dir/python_bindings.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/python_bindings.cpp.o -MF CMakeFiles/vgt.dir/python_bindings.cpp.o.d -o CMakeFiles/vgt.dir/python_bindings.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/python_bindings.cpp

vgt/CMakeFiles/vgt.dir/python_bindings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/python_bindings.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/python_bindings.cpp > CMakeFiles/vgt.dir/python_bindings.cpp.i

vgt/CMakeFiles/vgt.dir/python_bindings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/python_bindings.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/python_bindings.cpp -o CMakeFiles/vgt.dir/python_bindings.cpp.s

vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o: vgt/algo/VoronoiGraph.cpp
vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o -MF CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o.d -o CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/VoronoiGraph.cpp

vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/VoronoiGraph.cpp > CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.i

vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/VoronoiGraph.cpp -o CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.s

vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.o: vgt/algo/kernels.cpp
vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.o -MF CMakeFiles/vgt.dir/algo/kernels.cpp.o.d -o CMakeFiles/vgt.dir/algo/kernels.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/kernels.cpp

vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/algo/kernels.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/kernels.cpp > CMakeFiles/vgt.dir/algo/kernels.cpp.i

vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/algo/kernels.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/kernels.cpp -o CMakeFiles/vgt.dir/algo/kernels.cpp.s

vgt/CMakeFiles/vgt.dir/utils.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/utils.cpp.o: vgt/utils.cpp
vgt/CMakeFiles/vgt.dir/utils.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object vgt/CMakeFiles/vgt.dir/utils.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/utils.cpp.o -MF CMakeFiles/vgt.dir/utils.cpp.o.d -o CMakeFiles/vgt.dir/utils.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/utils.cpp

vgt/CMakeFiles/vgt.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/utils.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/utils.cpp > CMakeFiles/vgt.dir/utils.cpp.i

vgt/CMakeFiles/vgt.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/utils.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/utils.cpp -o CMakeFiles/vgt.dir/utils.cpp.s

vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.o: vgt/RandomEngine.cpp
vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.o -MF CMakeFiles/vgt.dir/RandomEngine.cpp.o.d -o CMakeFiles/vgt.dir/RandomEngine.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/RandomEngine.cpp

vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/RandomEngine.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/RandomEngine.cpp > CMakeFiles/vgt.dir/RandomEngine.cpp.i

vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/RandomEngine.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/RandomEngine.cpp -o CMakeFiles/vgt.dir/RandomEngine.cpp.s

vgt/CMakeFiles/vgt.dir/IndexSet.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/IndexSet.cpp.o: vgt/IndexSet.cpp
vgt/CMakeFiles/vgt.dir/IndexSet.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object vgt/CMakeFiles/vgt.dir/IndexSet.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/IndexSet.cpp.o -MF CMakeFiles/vgt.dir/IndexSet.cpp.o.d -o CMakeFiles/vgt.dir/IndexSet.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/IndexSet.cpp

vgt/CMakeFiles/vgt.dir/IndexSet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/IndexSet.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/IndexSet.cpp > CMakeFiles/vgt.dir/IndexSet.cpp.i

vgt/CMakeFiles/vgt.dir/IndexSet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/IndexSet.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/IndexSet.cpp -o CMakeFiles/vgt.dir/IndexSet.cpp.s

vgt/CMakeFiles/vgt.dir/KDTree.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/KDTree.cpp.o: vgt/KDTree.cpp
vgt/CMakeFiles/vgt.dir/KDTree.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object vgt/CMakeFiles/vgt.dir/KDTree.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/KDTree.cpp.o -MF CMakeFiles/vgt.dir/KDTree.cpp.o.d -o CMakeFiles/vgt.dir/KDTree.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/KDTree.cpp

vgt/CMakeFiles/vgt.dir/KDTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/KDTree.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/KDTree.cpp > CMakeFiles/vgt.dir/KDTree.cpp.i

vgt/CMakeFiles/vgt.dir/KDTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/KDTree.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/KDTree.cpp -o CMakeFiles/vgt.dir/KDTree.cpp.s

vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o: vgt/algo/cell_kernels.cpp
vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o -MF CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o.d -o CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/cell_kernels.cpp

vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/algo/cell_kernels.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/cell_kernels.cpp > CMakeFiles/vgt.dir/algo/cell_kernels.cpp.i

vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/algo/cell_kernels.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/cell_kernels.cpp -o CMakeFiles/vgt.dir/algo/cell_kernels.cpp.s

vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o: vgt/algo/kernels_gpu.cpp
vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o -MF CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o.d -o CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/kernels_gpu.cpp

vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/kernels_gpu.cpp > CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.i

vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/kernels_gpu.cpp -o CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.s

vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o: vgt/algo/VoronoiDensityEstimator.cpp
vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o -MF CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o.d -o CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/VoronoiDensityEstimator.cpp

vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/VoronoiDensityEstimator.cpp > CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.i

vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/algo/VoronoiDensityEstimator.cpp -o CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.s

vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o: vgt/extra/GabrielGraph.cpp
vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o -MF CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o.d -o CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/extra/GabrielGraph.cpp

vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/extra/GabrielGraph.cpp > CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.i

vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/extra/GabrielGraph.cpp -o CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.s

vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.o: vgt/CMakeFiles/vgt.dir/flags.make
vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.o: vgt/extra/KDE.cpp
vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.o: vgt/CMakeFiles/vgt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.o"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.o -MF CMakeFiles/vgt.dir/extra/KDE.cpp.o.d -o CMakeFiles/vgt.dir/extra/KDE.cpp.o -c /home/giovanni/Downloads/supplementary/rvde-code/vgt/extra/KDE.cpp

vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgt.dir/extra/KDE.cpp.i"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giovanni/Downloads/supplementary/rvde-code/vgt/extra/KDE.cpp > CMakeFiles/vgt.dir/extra/KDE.cpp.i

vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgt.dir/extra/KDE.cpp.s"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giovanni/Downloads/supplementary/rvde-code/vgt/extra/KDE.cpp -o CMakeFiles/vgt.dir/extra/KDE.cpp.s

# Object files for target vgt
vgt_OBJECTS = \
"CMakeFiles/vgt.dir/python_bindings.cpp.o" \
"CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o" \
"CMakeFiles/vgt.dir/algo/kernels.cpp.o" \
"CMakeFiles/vgt.dir/utils.cpp.o" \
"CMakeFiles/vgt.dir/RandomEngine.cpp.o" \
"CMakeFiles/vgt.dir/IndexSet.cpp.o" \
"CMakeFiles/vgt.dir/KDTree.cpp.o" \
"CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o" \
"CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o" \
"CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o" \
"CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o" \
"CMakeFiles/vgt.dir/extra/KDE.cpp.o"

# External object files for target vgt
vgt_EXTERNAL_OBJECTS =

vgt/vgt.so: vgt/CMakeFiles/vgt.dir/python_bindings.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/algo/VoronoiGraph.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/algo/kernels.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/utils.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/RandomEngine.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/IndexSet.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/KDTree.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/algo/cell_kernels.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/algo/kernels_gpu.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/algo/VoronoiDensityEstimator.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/extra/GabrielGraph.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/extra/KDE.cpp.o
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/build.make
vgt/vgt.so: libs/cnpy/libcnpy.so
vgt/vgt.so: libs/gpu/liblibgpu.so
vgt/vgt.so: /home/giovanni/anaconda3/envs/vgtenv/lib/libboost_python38.so.1.71.0
vgt/vgt.so: /home/giovanni/anaconda3/envs/vgtenv/lib/libboost_numpy38.so.1.71.0
vgt/vgt.so: /usr/lib/x86_64-linux-gnu/libpython3.10.so
vgt/vgt.so: /usr/lib/x86_64-linux-gnu/libz.so
vgt/vgt.so: libs/clew/liblibclew.so
vgt/vgt.so: /home/giovanni/anaconda3/envs/vgtenv/lib/libboost_python38.so.1.71.0
vgt/vgt.so: vgt/CMakeFiles/vgt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/giovanni/Downloads/supplementary/rvde-code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX shared module vgt.so"
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vgt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vgt/CMakeFiles/vgt.dir/build: vgt/vgt.so
.PHONY : vgt/CMakeFiles/vgt.dir/build

vgt/CMakeFiles/vgt.dir/clean:
	cd /home/giovanni/Downloads/supplementary/rvde-code/vgt && $(CMAKE_COMMAND) -P CMakeFiles/vgt.dir/cmake_clean.cmake
.PHONY : vgt/CMakeFiles/vgt.dir/clean

vgt/CMakeFiles/vgt.dir/depend: vgt/cl/voronoi_cl.h
	cd /home/giovanni/Downloads/supplementary/rvde-code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giovanni/Downloads/supplementary/rvde-code /home/giovanni/Downloads/supplementary/rvde-code/vgt /home/giovanni/Downloads/supplementary/rvde-code /home/giovanni/Downloads/supplementary/rvde-code/vgt /home/giovanni/Downloads/supplementary/rvde-code/vgt/CMakeFiles/vgt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vgt/CMakeFiles/vgt.dir/depend

