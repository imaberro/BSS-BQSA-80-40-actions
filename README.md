# BSS-BQSA

Work repository for "Swarm-Inspired Computing to Solve Binary Optimization Problems: A Backward Q-Learning Binarization Scheme Selector"

## Instructions

#### 1) Create PostgreSQL Database (file and scripts to create a data bases in local postgres are in file "Table_DB_mh_resultados.sql")

#### 2) Update "db_config.ini" with the credentials corresponding to the created database.

#### 3) Install the settings.py for install all packages and set up the envir

#### 4) Populate database with the experiments to be performed in "configure.py". You can follow the example with the "17.- configure (con 80a QL).py" file.

```
python configure.py
```

#### 5) Running experiments

```
python main.py
```
