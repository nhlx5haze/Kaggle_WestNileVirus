echo off
::untar model file to get modellasagne.dat and modelNNOctave.dat
.\extra\tar xvf %3
echo "**************  Predict from lasagne model ****************"
::This script will create submissionlasagna.tmp file
python ./src/predict.py %1 %2
echo "**************  Preprocessing Test data  ****************"
::This script will create preprocesstest.csv used by prdict.m script
python ./src/preprocess.py %1 test
echo "**************  Predict From NN Model  ****************"
octave-cli.exe ./src/predict.m
echo "**************  End  ****************"
del /f modellasagne.dat modelNNOctave.dat preprocesstest.csv submissionlasagna.tmp
