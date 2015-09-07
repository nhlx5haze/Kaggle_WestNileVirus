echo off
echo "**************  Build lasagne model ****************"
::This script will create a modellasagne.dat model file
python ./src/train.py %1 %2
echo "**************  Preprocessing training data  ****************"
::This script will create a preprocesstrain.csv used by train.m script
python ./src/preprocess.py %1 train
echo "**************  Build NN model  ****************"
::This script will create a modelNNOctave.dat model file
octave-cli.exe ./src/train.m
.\extra\tar -cf %3 modellasagne.dat modelNNOctave.dat
del /f modellasagne.dat modelNNOctave.dat preprocesstrain.csv
