This repository tries implements mathematical equations from linear algebra, probability and statistics and machine learning algorithms from scratch. 


To run use the root of the project 
for ex. python -m maths.probability.core.m01_bayes_theorem

Run normal tests:
python -m pytest -m "not performance and not scaling" -v

Run performance tests:
python -m pytest maths/linear_algebra/tests/a04_scaling.py -m performance -v

Run factorial/scaling tests:
python -m pytest maths/linear_algebra/tests/a04_scaling.py -m scaling -v