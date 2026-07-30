This repository tries implements mathematical equations from linear algebra, probability and statistics and machine learning algorithms from scratch. 


Running Visualizations

The visualization scripts are located inside the linear_algebra/visualization package. To run a visualization, execute it as a Python module from the project root so that package imports work correctly.

1. Activate the virtual environment

From the project root:

.venv\Scripts\activate

2. Navigate to the maths directory
cd maths

3. Run a visualization module

For example, to run the vector operations visualization:

python -m linear_algebra.visualization.v01_vector_ops


Running the file this way ensures that Python recognizes linear_algebra as a package and resolves internal imports correctly.

Output

Generated figures are saved automatically inside:

linear_algebra/
└── visualization/
    └── figures/


For example:

linear_algebra/visualization/figures/v01_vector_ops.png

Adding New Visualizations

When adding a new visualization:

Create a new Python file inside:
linear_algebra/visualization/

Save generated figures inside:
linear_algebra/visualization/figures/

Run the visualization using the module format:
python -m linear_algebra.visualization.<filename_without_py>


Example:

python -m linear_algebra.visualization.v02_matrix_ops


This will match the package structure you are using and also documents the important python -m workflow that avoids the import errors you encountered.


Kindly note while visualization uses ai for implementation, the library and its implementation of algorithms doesn't