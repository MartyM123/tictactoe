
# tictactoe AI

### files:

- ***\_\_init__.py*** - importing external modules like numpy, pickle etc.
- ***game.py*** - game logic of tictactoes
- ***math_utils.py*** - basic functions about math topic
- ***model*** - all with model loading, saving predicting, building
- ***train.py*** - all about learning
- ***utils.py*** - helping functions

your main file has to be outside folder tictactoe
tictactoe is python module : )

### function index:


- ***\_\_init__.py***
  - import modules

- ***game.py***
  - functions
    - `is_full`
    - `is_winner`
    - `is_legal`
    - `fight`
    - `execute`

- ***math_utils.py***
  - functions
    - `sigmoid`
    - `plain`
    - `add_random_n_places`
    - `get_max_exc`

- ***model***
  - classes
    - *dense*
      - `compile`
      - `compute`
      - `to_json`
      - `mutate`
    - *layer*
      - `__init__`
      - `__str__`
      - `compile`
      - `compute`
      - `mutate`
    - *model*
      - `__init__(self, layers=[]):`
      - `__str__(self)`
      - `__copy__`
      - `compute(self, input : np.ndarray)`
      - `compile(self)`
      - `save(self, name='save')`
      - `mutate(self)`
  - functions
    - `load_model`
    - `generate_random_models`

- ***train.py***
  - functions
    - `score`
    - `score_with_random`
    - `choose_parents`
    - `reproduce`
    - `one_cycle`

- ***utils.py***
  - classes
    - *counter*
      - `__init__`
      - `start`
      - `stop`
      - `count`
      - `reset`