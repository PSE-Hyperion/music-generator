# How to use Groove Panda

## How to run the program
The program is always run in Debug Mode. You can either press the Run and Debug button in the left sidebar and then the Start Debugging button, press the Run button in the top and select Start Debugging or press the F5 shortcut.

## The Command Line Interface
You interact with the program via the terminal in VSCode. There are commands with parameters, that you input to initiate different steps of the dataset-model-generation pipeline.

## The Commands
The available commands and their expected parameters are:
1. -help
2. -process dataset_id processed_id(new)
3. -train model_id/model_id(new) processed_id model_architecture_preset
4. -generate model_id input result_id(new)
5. -delete file/dataset/processed/model some_id/all
6. -exit
7. -config load/save/update/set/overwrite

"id" are given names (or simply the file names without extension). "(new)" indicates that the name should be a new one. Parameters seperated with "/" are both possible parameters.

Using tab while writing in the terminal will expose command and parameter completions. This should facilitate the use. The -config command doesn't have completion.

## How to configure the program and model (and more)
In `data/configs`, there should be a `config_final.json` JSON file. You can modify the values in this file to fit your needs. Alternatively, you could create a copy, modify this copy and change the CONFIG_NAME variable in `src/groove_panda/directories`. This will then use your new created config file. The specified configurations will be used automatically during runtime.

## The Pipeline
1. You will need a dataset. The dataset folder needs to be placed in the `data/datasets/raw` directory. The folder should contain no subfolders, just the songs in .mid/.midi format.
2. Now you can process the dataset, using the -process command in the terminal. The processed dataset will be saved with the provided name.
3. Now you create a model with the -train command. This creates a new model and immediatly trains it with the given processed dataset and will save it afterwards with the provided name.
4. Finally you can generate using the -generate command. For this, a midi file in the `data/generation/input` is required. You can now use this input to generate a new song. This song will be saved in `data/generation/output` with the provided name.

